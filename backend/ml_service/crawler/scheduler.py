"""Crawl scheduler — orchestrate multi-provider crawling.

Runs all registered providers, merges results, deduplicates, saves.
"""

from __future__ import annotations

import logging
import time

from ml_service.crawler.base import RawJob
from ml_service.crawler.factory import get_provider, list_providers
from ml_service.crawler.storage import deduplicate, save_raw_jobs

logger = logging.getLogger(__name__)


class CrawlScheduler:
    """Run multiple crawl providers and merge results.

    Usage:
        scheduler = CrawlScheduler(providers=["jobspy", "remotive"])
        results = scheduler.crawl_all(
            queries=["python developer", "react developer"],
            results_per_query=50,
        )
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        **provider_kwargs,
    ) -> None:
        if providers is None:
            providers = list_providers()
        self._provider_names = providers
        self._provider_kwargs = provider_kwargs

    def crawl_all(
        self,
        queries: list[str],
        results_per_query: int = 50,
        location: str = "",
        save_path: str | None = None,
    ) -> list[RawJob]:
        """Crawl all queries across all providers, merge and dedup.

        Returns list of unique RawJob.
        """
        t_start = time.time()
        all_jobs: list[RawJob] = []

        for provider_name in self._provider_names:
            try:
                provider = get_provider(provider_name, **self._provider_kwargs)
                logger.info("=== Provider: %s ===", provider_name)

                for query in queries:
                    try:
                        jobs = provider.fetch(
                            search_term=query,
                            location=location,
                            results_wanted=results_per_query,
                        )
                        all_jobs.extend(jobs)
                        logger.info("  %s → %s: %d jobs", provider_name, query, len(jobs))
                    except Exception as e:
                        logger.error("  %s → %s: FAILED: %s", provider_name, query, e)

            except Exception as e:
                logger.error("Failed to init provider %s: %s", provider_name, e)

        # Dedup
        before = len(all_jobs)
        all_jobs = deduplicate(all_jobs)
        logger.info(
            "Total: %d raw → %d unique (%.1fs)",
            before, len(all_jobs), time.time() - t_start,
        )

        # Save if path provided
        if save_path:
            written = save_raw_jobs(all_jobs, save_path)
            logger.info("Saved %d jobs to %s", written, save_path)

        return all_jobs

    def crawl_provider(
        self,
        provider_name: str,
        queries: list[str],
        results_per_query: int = 50,
        location: str = "",
    ) -> list[RawJob]:
        """Crawl from a single provider."""
        provider = get_provider(provider_name, **self._provider_kwargs)
        jobs: list[RawJob] = []

        for query in queries:
            try:
                batch = provider.fetch(
                    search_term=query,
                    location=location,
                    results_wanted=results_per_query,
                )
                jobs.extend(batch)
            except Exception as e:
                logger.error("%s → %s: %s", provider_name, query, e)

        return deduplicate(jobs)
