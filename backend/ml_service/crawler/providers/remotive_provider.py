"""CrawlProvider backed by Remotive API.

Free, no auth needed. Remote tech jobs only.
https://remotive.com/api/remote-jobs
"""

from __future__ import annotations

import logging
from datetime import datetime

import requests

from ml_service.crawler.base import CrawlProvider, RawJob

logger = logging.getLogger(__name__)

_API_URL = "https://remotive.com/api/remote-jobs"

# Remotive category slugs for IT
_IT_CATEGORIES = [
    "software-dev",
    "data",
    "devops",
    "qa",
    "product",
]


class RemotiveProvider(CrawlProvider):
    """Remotive API provider — free, remote tech jobs.

    Usage:
        provider = RemotiveProvider()
        jobs = provider.fetch("python developer", results_wanted=50)
    """

    @property
    def name(self) -> str:
        return "remotive"

    def fetch(
        self,
        search_term: str,
        location: str = "",
        results_wanted: int = 100,
        **kwargs,
    ) -> list[RawJob]:
        category = kwargs.get("category", "software-dev")

        try:
            params = {"search": search_term, "limit": results_wanted}
            if category:
                params["category"] = category

            resp = requests.get(_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            jobs_data = data.get("jobs", [])
            logger.info("Remotive returned %d jobs for '%s'", len(jobs_data), search_term)

            results: list[RawJob] = []
            for item in jobs_data[:results_wanted]:
                job = self._to_raw_job(item)
                if job:
                    results.append(job)

            return results

        except Exception as e:
            logger.error("Remotive fetch error: %s", e)
            return []

    @staticmethod
    def _to_raw_job(item: dict) -> RawJob | None:
        title = item.get("title", "").strip()
        description = item.get("description", "").strip()
        if not title or not description:
            return None

        # Clean HTML from description
        import re
        description = re.sub(r"<[^>]+>", " ", description)
        description = re.sub(r"\s+", " ", description).strip()

        company = item.get("company_name", "")
        location = item.get("candidate_required_location", "Worldwide")

        salary = item.get("salary", "")
        salary_min, salary_max = _parse_salary(salary)

        date_str = item.get("publication_date")
        date_posted = None
        if date_str:
            try:
                date_posted = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        tags = item.get("tags", [])

        return RawJob(
            source="remotive",
            source_url=item.get("url", ""),
            title=title,
            company=company,
            location=location,
            description=description[:5000],
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency="USD",
            date_posted=date_posted,
            raw_skills=tuple(tags) if tags else (),
        )


def _parse_salary(salary_str: str) -> tuple[float | None, float | None]:
    """Parse salary string like '$50,000 - $80,000' or '50000-80000'."""
    if not salary_str:
        return None, None
    import re
    numbers = re.findall(r"[\d,]+", salary_str.replace(",", ""))
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    elif len(numbers) == 1:
        return float(numbers[0]), float(numbers[0])
    return None, None
