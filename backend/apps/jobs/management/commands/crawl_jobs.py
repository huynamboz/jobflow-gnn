"""Crawl jobs from providers and save to DB.

Usage:
    python manage.py crawl_jobs --provider jobspy --query "react developer" --results 50
    python manage.py crawl_jobs --provider linkedin --query "developer" --location "Vietnam" --results 100
    python manage.py crawl_jobs --all --results 50
"""

from django.core.management.base import BaseCommand

from apps.jobs.services import JobService
from ml_service.crawler import CrawlScheduler, get_provider, list_providers


DEFAULT_QUERIES = [
    "software engineer",
    "python developer",
    "frontend developer",
    "backend developer",
    "fullstack developer",
    "react developer",
    "data engineer",
    "devops engineer",
]


class Command(BaseCommand):
    help = "Crawl jobs from providers and save to database"

    def add_arguments(self, parser):
        parser.add_argument("--provider", type=str, default="jobspy", help=f"Provider: {', '.join(list_providers())}")
        parser.add_argument("--query", type=str, help="Search query (single). If not set, uses default queries.")
        parser.add_argument("--location", type=str, default="", help="Location filter")
        parser.add_argument("--results", type=int, default=50, help="Results per query")
        parser.add_argument("--all", action="store_true", help="Use all available providers")
        parser.add_argument("--save-file", action="store_true", help="Also save to data/raw_jobs.jsonl")

    def handle(self, *args, **options):
        provider_name = options["provider"]
        query = options["query"]
        location = options["location"]
        results_wanted = options["results"]
        use_all = options["all"]
        save_file = options["save_file"]

        queries = [query] if query else DEFAULT_QUERIES
        providers = list_providers() if use_all else [provider_name]

        self.stdout.write(f"Providers: {providers}")
        self.stdout.write(f"Queries: {queries}")
        self.stdout.write(f"Results per query: {results_wanted}")

        # Crawl
        all_jobs = []
        for pname in providers:
            try:
                kwargs = {}
                if save_file and pname == "linkedin":
                    kwargs["save_path"] = "data/raw_jobs.jsonl"
                p = get_provider(pname, **kwargs)
                for q in queries:
                    try:
                        jobs = p.fetch(q, location=location, results_wanted=results_wanted)
                        all_jobs.extend(jobs)
                        self.stdout.write(f"  {pname} → {q}: {len(jobs)} jobs")
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f"  {pname} → {q}: FAILED ({e})"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  Provider {pname} init failed: {e}"))

        self.stdout.write(f"\nTotal crawled: {len(all_jobs)}")

        # Save to DB
        service = JobService()
        stats = service.save_raw_jobs_batch(all_jobs)

        self.stdout.write(self.style.SUCCESS(
            f"DB: {stats['created']} created, {stats['skipped']} duplicates, {stats['failed']} failed"
        ))
