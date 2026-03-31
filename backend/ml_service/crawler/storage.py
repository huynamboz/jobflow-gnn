"""JSON-lines storage for crawled jobs with multi-layer deduplication.

Dedup layers:
  1. URL exact match (same listing from same source)
  2. Fingerprint match (same job across different sources/URLs)
     fingerprint = hash(normalize(title) + normalize(company) + city)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from ml_service.crawler.base import RawJob


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

_SENIORITY_PREFIXES = re.compile(
    r"\b(sr\.?|jr\.?|senior|junior|lead|staff|principal|intern)\b", re.I
)
_COMPANY_SUFFIXES = re.compile(
    r"\b(inc\.?|llc\.?|ltd\.?|corp\.?|co\.?|company|group|gmbh|plc)\b", re.I
)
_WHITESPACE = re.compile(r"\s+")

# Common abbreviations in job titles
_TITLE_ABBREVIATIONS: dict[str, str] = {
    "dev": "developer",
    "devs": "developers",
    "eng": "engineer",
    "engg": "engineer",
    "mgr": "manager",
    "admin": "administrator",
    "swe": "software engineer",
    "sde": "software development engineer",
    "qa": "quality assurance",
    "ui": "user interface",
    "ux": "user experience",
    "fe": "frontend",
    "be": "backend",
    "fs": "fullstack",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "db": "database",
    "sys": "system",
}


def _normalize_title(title: str) -> str:
    """Lowercase, strip seniority, expand abbreviations, collapse whitespace."""
    t = title.lower().strip()
    t = _SENIORITY_PREFIXES.sub(" ", t)
    t = re.sub(r"[.\-,/]", " ", t)  # strip punctuation
    words = t.split()
    words = [_TITLE_ABBREVIATIONS.get(w, w) for w in words]
    t = " ".join(words)
    t = _WHITESPACE.sub(" ", t).strip()
    return t


def _normalize_company(company: str) -> str:
    """Lowercase, strip legal suffixes and punctuation."""
    c = company.lower().strip()
    c = _COMPANY_SUFFIXES.sub(" ", c)
    c = re.sub(r"[.\-,]", " ", c)
    c = _WHITESPACE.sub(" ", c).strip()
    return c


def _extract_city(location: str) -> str:
    """Extract first part before comma as city."""
    if not location:
        return ""
    parts = location.split(",")
    return parts[0].strip().lower()


def compute_fingerprint(job: RawJob) -> str:
    """Stable fingerprint from normalized title + company + city.

    Same job posted on Indeed and Adzuna → same fingerprint.
    "Senior Python Developer @ Google Inc., Mountain View, CA"
    "Sr. Python Dev @ Google, Mountain View"
    → same fingerprint
    """
    title = _normalize_title(job.title)
    company = _normalize_company(job.company)
    city = _extract_city(job.location)
    raw = f"{title}|{company}|{city}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate(jobs: list[RawJob]) -> list[RawJob]:
    """Remove duplicates using 2 layers: URL + fingerprint.

    Layer 1: exact source_url match → skip (same listing, same source)
    Layer 2: fingerprint match → skip (same job, different source/URL)
    """
    seen_urls: set[str] = set()
    seen_fps: set[str] = set()
    result: list[RawJob] = []

    for job in jobs:
        # Layer 1: URL dedup
        if job.source_url and job.source_url in seen_urls:
            continue

        # Layer 2: Fingerprint dedup
        fp = compute_fingerprint(job)
        if fp in seen_fps:
            continue

        if job.source_url:
            seen_urls.add(job.source_url)
        seen_fps.add(fp)
        result.append(job)

    return result


# ---------------------------------------------------------------------------
# JSONL storage
# ---------------------------------------------------------------------------


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
        "company_logo_url": job.company_logo_url,
        "company_url": job.company_url,
        "job_type": job.job_type,
        "applicant_count": job.applicant_count,
        "fingerprint": compute_fingerprint(job),
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
        company_logo_url=obj.get("company_logo_url", ""),
        company_url=obj.get("company_url", ""),
        job_type=obj.get("job_type", ""),
        applicant_count=obj.get("applicant_count", ""),
    )
