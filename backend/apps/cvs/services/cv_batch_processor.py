"""Background batch processor for CV re-extraction."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_cancel_events: dict[int, threading.Event] = {}
_lock = threading.Lock()

WORKERS = 10


def start_batch(batch_id: int) -> None:
    cancel_event = threading.Event()
    with _lock:
        _cancel_events[batch_id] = cancel_event
    thread = threading.Thread(target=_run, args=(batch_id, cancel_event), daemon=True)
    thread.start()


def cancel_batch(batch_id: int) -> bool:
    with _lock:
        event = _cancel_events.get(batch_id)
    if event:
        event.set()
        return True
    return False


def is_running(batch_id: int) -> bool:
    with _lock:
        return batch_id in _cancel_events


def _apply_result_to_cv(cv_id: int, result_dict: dict) -> None:
    """Update CV record and its skills from extraction result."""
    from django.db import transaction
    from apps.cvs.models import CV, CVSkill
    from apps.cvs.services.cv_service import CVService, _SENIORITY_DEFAULT_YEARS
    from apps.skills.services import SkillService
    from ml_service.data.skill_normalization import SkillNormalizer

    normalizer = SkillNormalizer()
    skill_svc = SkillService()

    seniority_raw = result_dict.get("seniority", -1)
    experience_years = float(result_dict.get("experience_years") or 0)

    if seniority_raw >= 0:
        seniority = seniority_raw
    else:
        seniority = CVService._years_to_seniority(experience_years)

    if experience_years == 0 and seniority >= 2:
        experience_years = _SENIORITY_DEFAULT_YEARS.get(seniority, 0.0)

    with transaction.atomic():
        CV.objects.filter(id=cv_id).update(
            candidate_name=result_dict.get("name", ""),
            seniority=seniority,
            experience_years=experience_years,
            education=result_dict.get("education", 2),
            role_category=result_dict.get("role_category", "other"),
            work_experience=result_dict.get("work_experience") or [],
        )

        CVSkill.objects.filter(cv_id=cv_id).delete()
        for s in result_dict.get("skills") or []:
            canonical = normalizer.normalize(s.get("name", ""))
            if not canonical:
                continue
            skill = skill_svc.get_or_create(canonical)
            if skill:
                proficiency = max(1, min(5, int(s.get("proficiency") or 3)))
                CVSkill.objects.get_or_create(
                    cv_id=cv_id, skill=skill,
                    defaults={"proficiency": proficiency},
                )


def _process_one(record_id: int, cv_id: int, raw_text: str) -> tuple[dict | None, str | None]:
    """Call LLM for a single CV record. Returns (result_dict, error_msg)."""
    from dataclasses import asdict
    from apps.cvs.services.llm_cv_extractor import extract
    try:
        result = extract(raw_text)
        return asdict(result), None
    except Exception as exc:
        return None, str(exc)[:500]


def _run(batch_id: int, cancel_event: threading.Event) -> None:
    import django
    django.setup()

    from django.db.models import F
    from apps.cvs.models import CVExtractionBatch, CVExtractionRecord

    try:
        CVExtractionBatch.objects.filter(id=batch_id).update(status=CVExtractionBatch.STATUS_RUNNING)

        records = list(
            CVExtractionRecord.objects.filter(
                batch_id=batch_id, status=CVExtractionRecord.STATUS_PENDING
            ).select_related("cv").order_by("cv_id")
        )

        # Separate out records with no raw_text immediately
        valid, empty = [], []
        for r in records:
            if r.cv.raw_text.strip():
                valid.append(r)
            else:
                empty.append(r)

        if empty:
            empty_ids = [r.id for r in empty]
            CVExtractionRecord.objects.filter(id__in=empty_ids).update(
                status=CVExtractionRecord.STATUS_ERROR,
                error_msg="CV has no raw_text",
            )
            CVExtractionBatch.objects.filter(id=batch_id).update(
                error_count=F("error_count") + len(empty)
            )

        logger.info("CV Batch #%d: %d records (%d valid, %d empty)", batch_id, len(records), len(valid), len(empty))

        pool = ThreadPoolExecutor(max_workers=WORKERS)
        future_to_record: dict = {}
        cancelled = False
        try:
            CHUNK = WORKERS * 2
            for i in range(0, len(valid), CHUNK):
                if cancel_event.is_set():
                    cancelled = True
                    break
                for r in valid[i:i + CHUNK]:
                    CVExtractionRecord.objects.filter(id=r.id).update(
                        status=CVExtractionRecord.STATUS_PROCESSING
                    )
                    future_to_record[pool.submit(_process_one, r.id, r.cv_id, r.cv.raw_text)] = r

            for future in as_completed(future_to_record):
                if cancel_event.is_set():
                    cancelled = True
                    for f in future_to_record:
                        f.cancel()
                    break

                record = future_to_record[future]
                result_dict, error_msg = future.result()

                if result_dict is not None:
                    CVExtractionRecord.objects.filter(id=record.id).update(
                        status=CVExtractionRecord.STATUS_DONE,
                        result=result_dict,
                    )
                    _apply_result_to_cv(record.cv_id, result_dict)
                    CVExtractionBatch.objects.filter(id=batch_id).update(
                        done_count=F("done_count") + 1
                    )
                else:
                    CVExtractionRecord.objects.filter(id=record.id).update(
                        status=CVExtractionRecord.STATUS_ERROR,
                        error_msg=error_msg,
                    )
                    CVExtractionBatch.objects.filter(id=batch_id).update(
                        error_count=F("error_count") + 1
                    )
                    logger.warning("CV #%d in batch %d failed: %s", record.cv_id, batch_id, error_msg)
        finally:
            pool.shutdown(wait=not cancelled)

        final = (
            CVExtractionBatch.STATUS_CANCELLED if cancel_event.is_set()
            else CVExtractionBatch.STATUS_DONE
        )
        final_done = CVExtractionRecord.objects.filter(batch_id=batch_id, status=CVExtractionRecord.STATUS_DONE).count()
        final_error = CVExtractionRecord.objects.filter(batch_id=batch_id, status=CVExtractionRecord.STATUS_ERROR).count()
        CVExtractionBatch.objects.filter(id=batch_id).update(
            status=final, done_count=final_done, error_count=final_error
        )

    except Exception as exc:
        logger.error("CV batch %d crashed: %s", batch_id, exc)
        CVExtractionBatch.objects.filter(id=batch_id).update(status=CVExtractionBatch.STATUS_ERROR)
    finally:
        with _lock:
            _cancel_events.pop(batch_id, None)
