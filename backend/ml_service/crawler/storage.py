"""Simple JSON-lines storage for crawled jobs.

Stores RawJob as JSONL (one JSON object per line) for easy append + stream.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ml_service.crawler.base import RawJob


def save_raw_jobs(jobs: list[RawJob], path: Path | str) -> int:
    """Append RawJobs to a JSONL file. Returns number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for job in jobs:
            obj = _raw_job_to_dict(job)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    return written


def load_raw_jobs(path: Path | str) -> list[RawJob]:
    """Load all RawJobs from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    jobs: list[RawJob] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            jobs.append(_dict_to_raw_job(obj))
    return jobs


def deduplicate(jobs: list[RawJob]) -> list[RawJob]:
    """Remove duplicates by source_url."""
    seen: set[str] = set()
    result: list[RawJob] = []
    for job in jobs:
        key = job.source_url
        if key and key not in seen:
            seen.add(key)
            result.append(job)
        elif not key:
            result.append(job)  # keep jobs without URL
    return result


def _raw_job_to_dict(job: RawJob) -> dict:
    return {
        "source": job.source,
        "source_url": job.source_url,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "description": job.description,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "salary_currency": job.salary_currency,
        "date_posted": job.date_posted.isoformat() if job.date_posted else None,
        "seniority_hint": job.seniority_hint,
        "raw_skills": list(job.raw_skills),
    }


def _dict_to_raw_job(obj: dict) -> RawJob:
    date_posted = None
    if obj.get("date_posted"):
        try:
            date_posted = datetime.fromisoformat(obj["date_posted"])
        except (ValueError, TypeError):
            pass
    return RawJob(
        source=obj.get("source", ""),
        source_url=obj.get("source_url", ""),
        title=obj.get("title", ""),
        company=obj.get("company", ""),
        location=obj.get("location", ""),
        description=obj.get("description", ""),
        salary_min=obj.get("salary_min"),
        salary_max=obj.get("salary_max"),
        salary_currency=obj.get("salary_currency", "USD"),
        date_posted=date_posted,
        seniority_hint=obj.get("seniority_hint"),
        raw_skills=tuple(obj.get("raw_skills", [])),
    )
