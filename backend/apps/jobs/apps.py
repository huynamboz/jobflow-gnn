from django.apps import AppConfig


class JobsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.jobs'

    def ready(self):
        _mark_orphaned_batches_error()


def _mark_orphaned_batches_error():
    """On startup: find JD batches stuck in 'running' and mark as error (user must Resume manually)."""
    try:
        from apps.jobs.models import JDExtractionBatch, JDExtractionRecord

        orphaned = list(JDExtractionBatch.objects.filter(status=JDExtractionBatch.STATUS_RUNNING))
        if not orphaned:
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Found %d orphaned JD batch(es) — marking as error.", len(orphaned))

        for batch in orphaned:
            stuck = JDExtractionRecord.objects.filter(
                batch=batch, status=JDExtractionRecord.STATUS_PROCESSING
            )
            count = stuck.update(status=JDExtractionRecord.STATUS_PENDING)
            if count:
                logger.info("JD Batch #%d: reset %d stuck records to pending.", batch.id, count)

            JDExtractionBatch.objects.filter(id=batch.id).update(
                status=JDExtractionBatch.STATUS_ERROR
            )
            logger.info("JD Batch #%d marked as error (server was restarted).", batch.id)

    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Failed to mark orphaned JD batches: %s", exc)
