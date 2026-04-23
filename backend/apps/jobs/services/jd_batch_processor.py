"""Background batch processor for JD extraction."""

from __future__ import annotations

import atexit
import hashlib
import logging
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

logger = logging.getLogger(__name__)

_cancel_events: dict[int, threading.Event] = {}
_threads: dict[int, threading.Thread] = {}
_active_pools: dict[int, ThreadPoolExecutor] = {}
_lock = threading.Lock()

WORKERS = 3


def _shutdown_all() -> None:
    """Cancel all running JD batches — called on SIGTERM/SIGINT/atexit."""
    with _lock:
        events = list(_cancel_events.values())
        pools = list(_active_pools.values())
    for event in events:
        event.set()
    for pool in pools:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


atexit.register(_shutdown_all)

try:
    _prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum, frame) -> None:
        _shutdown_all()
        if callable(_prev_sigterm) and _prev_sigterm not in (signal.SIG_DFL, signal.SIG_IGN):
            _prev_sigterm(signum, frame)

    signal.signal(signal.SIGTERM, _signal_handler)
except (OSError, ValueError):
    pass


def start_batch(batch_id: int, workers: int = WORKERS) -> None:
    # Stop any existing thread for this batch before starting a new one
    with _lock:
        old_event = _cancel_events.get(batch_id)
        old_thread = _threads.get(batch_id)

    if old_event and not old_event.is_set():
        old_event.set()
    if old_thread and old_thread.is_alive():
        old_thread.join(timeout=10)

    cancel_event = threading.Event()
    thread = threading.Thread(target=_run, args=(batch_id, cancel_event, workers), daemon=True)
    with _lock:
        _cancel_events[batch_id] = cancel_event
        _threads[batch_id] = thread
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
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _process_one(record_id: int, combined_text: str) -> tuple[dict | None, str | None]:
    """Call LLM for a single record. Returns (result_dict, error_msg)."""
    from apps.jobs.services.llm_jd_extractor import extract
    try:
        result = extract(combined_text)
        return asdict(result), None
    except Exception as exc:
        return None, str(exc)[:500]


def _run(batch_id: int, cancel_event: threading.Event, workers: int = WORKERS) -> None:
    import django
    django.setup()

    from django.db.models import F
    from apps.jobs.models import JDExtractionBatch, JDExtractionRecord

    try:
        JDExtractionBatch.objects.filter(id=batch_id).update(status=JDExtractionBatch.STATUS_RUNNING)

        records = list(
            JDExtractionRecord.objects.filter(
                batch_id=batch_id, status=JDExtractionRecord.STATUS_PENDING
            ).order_by("index")
        )

        # Group by content hash — process only unique content, copy result to dupes
        hash_to_primary: dict[str, JDExtractionRecord] = {}
        hash_to_dupes: dict[str, list[JDExtractionRecord]] = {}
        for r in records:
            h = content_hash(r.combined_text)
            if not r.content_hash:
                JDExtractionRecord.objects.filter(id=r.id).update(content_hash=h)
            if h not in hash_to_primary:
                hash_to_primary[h] = r
                hash_to_dupes[h] = []
            else:
                hash_to_dupes[h].append(r)

        unique_records = list(hash_to_primary.values())
        logger.info(
            "JD Batch #%d: %d records, %d unique after dedup",
            batch_id, len(records), len(unique_records),
        )

        pool = ThreadPoolExecutor(max_workers=workers)
        with _lock:
            _active_pools[batch_id] = pool
        future_to_record: dict = {}
        cancelled = False
        try:
            # Submit in chunks so cancel can stop queued work quickly
            CHUNK = workers * 2
            for i in range(0, len(unique_records), CHUNK):
                if cancel_event.is_set():
                    cancelled = True
                    break
                for r in unique_records[i:i + CHUNK]:
                    if cancel_event.is_set():
                        cancelled = True
                        break
                    JDExtractionRecord.objects.filter(id=r.id).update(
                        status=JDExtractionRecord.STATUS_PROCESSING
                    )
                    future_to_record[pool.submit(_process_one, r.id, r.combined_text)] = r
                if cancelled:
                    break

            for future in as_completed(future_to_record):
                if cancel_event.is_set():
                    cancelled = True
                    for f in future_to_record:
                        f.cancel()
                    break

                record = future_to_record[future]
                h = content_hash(record.combined_text)
                result_dict, error_msg = future.result()

                if result_dict is not None:
                    JDExtractionRecord.objects.filter(id=record.id).update(
                        status=JDExtractionRecord.STATUS_DONE,
                        result=result_dict,
                    )
                    JDExtractionBatch.objects.filter(id=batch_id).update(
                        done_count=F("done_count") + 1
                    )
                    dupes = hash_to_dupes.get(h, [])
                    if dupes:
                        dupe_ids = [d.id for d in dupes]
                        JDExtractionRecord.objects.filter(id__in=dupe_ids).update(
                            status=JDExtractionRecord.STATUS_DONE,
                            result=result_dict,
                        )
                        JDExtractionBatch.objects.filter(id=batch_id).update(
                            done_count=F("done_count") + len(dupes)
                        )
                else:
                    JDExtractionRecord.objects.filter(id=record.id).update(
                        status=JDExtractionRecord.STATUS_ERROR,
                        error_msg=error_msg,
                    )
                    JDExtractionBatch.objects.filter(id=batch_id).update(
                        error_count=F("error_count") + 1
                    )
                    logger.warning("Record %d (batch %d) failed: %s", record.id, batch_id, error_msg)
        finally:
            with _lock:
                _active_pools.pop(batch_id, None)
            pool.shutdown(wait=False, cancel_futures=True) if cancelled else pool.shutdown(wait=True)

        final = (
            JDExtractionBatch.STATUS_CANCELLED if cancel_event.is_set()
            else JDExtractionBatch.STATUS_DONE
        )
        final_done = JDExtractionRecord.objects.filter(batch_id=batch_id, status=JDExtractionRecord.STATUS_DONE).count()
        final_error = JDExtractionRecord.objects.filter(batch_id=batch_id, status=JDExtractionRecord.STATUS_ERROR).count()
        JDExtractionBatch.objects.filter(id=batch_id).update(
            status=final, done_count=final_done, error_count=final_error
        )

    except Exception as exc:
        logger.error("JD Batch %d crashed: %s", batch_id, exc)
        JDExtractionBatch.objects.filter(id=batch_id).update(status=JDExtractionBatch.STATUS_ERROR)
    finally:
        with _lock:
            _cancel_events.pop(batch_id, None)
            _threads.pop(batch_id, None)
