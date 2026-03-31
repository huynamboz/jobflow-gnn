"""Import jobs from JSONL file to DB.

Usage: python manage.py import_jobs [--file data/raw_jobs.jsonl]
"""

from django.core.management.base import BaseCommand

from apps.jobs.services import JobService
from ml_service.crawler.storage import deduplicate, load_raw_jobs


class Command(BaseCommand):
    help = "Import crawled jobs from JSONL file to database"

    def add_arguments(self, parser):
        parser.add_argument("--file", default="data/raw_jobs.jsonl", help="Path to JSONL file")

    def handle(self, *args, **options):
        path = options["file"]
        self.stdout.write(f"Loading jobs from {path}...")

        raws = load_raw_jobs(path)
        raws = deduplicate(raws)
        self.stdout.write(f"Loaded {len(raws)} unique jobs")

        service = JobService()
        stats = service.save_raw_jobs_batch(raws)

        self.stdout.write(self.style.SUCCESS(
            f"Done: {stats['created']} created, {stats['skipped']} skipped, {stats['failed']} failed"
        ))
