"""Background batch processor for JD extraction."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import asdict

logger = logging.getLogger(__name__)

_cancel_events: dict[int, threading.Event] = {}
_lock = threading.Lock()


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


def content_hash(text: str) -> str:
    """MD5 of combined_text — used to detect duplicate rows within a batch."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _run(batch_id: int, cancel_event: threading.Event) -> None:
    import django
    django.setup()

    from django.db.models import F
    from apps.jobs.models import JDExtractionBatch, JDExtractionRecord
    from apps.jobs.services.llm_jd_extractor import extract

    try:
        JDExtractionBatch.objects.filter(id=batch_id).update(status=JDExtractionBatch.STATUS_RUNNING)

        records_qs = JDExtractionRecord.objects.filter(
            batch_id=batch_id, status=JDExtractionRecord.STATUS_PENDING
        ).order_by("index")

        # Cache of hash → result for already-processed records in this batch
        seen_hashes: dict[str, dict] = {}

        for record in records_qs.iterator(chunk_size=50):
            if cancel_event.is_set():
                break

            h = content_hash(record.combined_text)

            # Stamp the hash so we can query duplicates later
            if not record.content_hash:
                JDExtractionRecord.objects.filter(id=record.id).update(content_hash=h)

            # Dedup: if this content was already extracted in this batch, reuse the result
            if h in seen_hashes:
                logger.info(
                    "Record %d (batch %d): duplicate content — reusing result from earlier record",
                    record.id, batch_id,
                )
                JDExtractionRecord.objects.filter(id=record.id).update(
                    status=JDExtractionRecord.STATUS_DONE,
                    result=seen_hashes[h],
                )
                JDExtractionBatch.objects.filter(id=batch_id).update(done_count=F("done_count") + 1)
                continue

            JDExtractionRecord.objects.filter(id=record.id).update(
                status=JDExtractionRecord.STATUS_PROCESSING
            )

            try:
                result = extract(record.combined_text)
                result_dict = asdict(result)
                JDExtractionRecord.objects.filter(id=record.id).update(
                    status=JDExtractionRecord.STATUS_DONE,
                    result=result_dict,
                )
                JDExtractionBatch.objects.filter(id=batch_id).update(done_count=F("done_count") + 1)
                seen_hashes[h] = result_dict
            except Exception as exc:
                JDExtractionRecord.objects.filter(id=record.id).update(
                    status=JDExtractionRecord.STATUS_ERROR,
                    error_msg=str(exc)[:500],
                )
                JDExtractionBatch.objects.filter(id=batch_id).update(error_count=F("error_count") + 1)
                logger.warning("Record %d in batch %d failed: %s", record.id, batch_id, exc)

        final = (
            JDExtractionBatch.STATUS_CANCELLED if cancel_event.is_set()
            else JDExtractionBatch.STATUS_DONE
        )
        JDExtractionBatch.objects.filter(id=batch_id).update(status=final)

    except Exception as exc:
        logger.error("Batch %d crashed: %s", batch_id, exc)
        JDExtractionBatch.objects.filter(id=batch_id).update(status=JDExtractionBatch.STATUS_ERROR)
    finally:
        with _lock:
            _cancel_events.pop(batch_id, None)
