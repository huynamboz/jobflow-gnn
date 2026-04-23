from django.apps import AppConfig


class CvsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.cvs'

    def ready(self):
        _resume_orphaned_batches()


def _resume_orphaned_batches():
    """On startup: find batches stuck in 'running' (server was restarted mid-batch).
    Reset processing records back to pending, then resume each batch.
    """
    try:
        from apps.cvs.models import CVExtractionBatch, CVExtractionRecord
        from apps.cvs.services.cv_batch_processor import start_batch

        orphaned = list(CVExtractionBatch.objects.filter(status=CVExtractionBatch.STATUS_RUNNING))
        if not orphaned:
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Found %d orphaned running batch(es) — resuming.", len(orphaned))

        for batch in orphaned:
            # Records stuck at "processing" were mid-flight when server died → reset to pending
            stuck = CVExtractionRecord.objects.filter(
                batch=batch, status=CVExtractionRecord.STATUS_PROCESSING
            )
            count = stuck.update(status=CVExtractionRecord.STATUS_PENDING)
            if count:
                logger.info("Batch #%d: reset %d stuck records to pending.", batch.id, count)

            start_batch(batch.id)
            logger.info("Batch #%d resumed.", batch.id)

    except Exception as exc:
        # Never crash startup — just log
        import logging
        logging.getLogger(__name__).error("Failed to resume orphaned batches: %s", exc)
