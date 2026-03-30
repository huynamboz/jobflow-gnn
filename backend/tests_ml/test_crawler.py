"""Tests for crawler module (base, factory, skill_extractor, storage)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ml_service.crawler.base import CrawlProvider, RawJob
from ml_service.crawler.factory import get_provider, register_provider
from ml_service.crawler.skill_extractor import SkillExtractor
from ml_service.crawler.storage import (
    deduplicate,
    load_raw_jobs,
    save_raw_jobs,
)
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import SeniorityLevel


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


@pytest.fixture
def extractor(normalizer):
    return SkillExtractor(normalizer)


def _make_raw_job(**overrides) -> RawJob:
    defaults = {
        "source": "test",
        "source_url": "https://example.com/job/1",
        "title": "Senior Python Developer",
        "company": "Acme Corp",
        "location": "Remote",
        "description": (
            "We are looking for a senior Python developer with experience in "
            "Django, PostgreSQL, Docker, and AWS. Must know REST APIs and Git. "
            "Experience with React is a plus."
        ),
        "salary_min": 80000.0,
        "salary_max": 120000.0,
        "salary_currency": "USD",
    }
    defaults.update(overrides)
    return RawJob(**defaults)


# ── RawJob ───────────────────────────────────────────────────────────────────


def test_raw_job_frozen():
    job = _make_raw_job()
    with pytest.raises(AttributeError):
        job.title = "changed"


def test_raw_job_defaults():
    job = RawJob(source="x", source_url="", title="t", company="c", location="l", description="d")
    assert job.salary_min is None
    assert job.salary_currency == "USD"
    assert job.raw_skills == ()


# ── Factory ──────────────────────────────────────────────────────────────────


def test_factory_jobspy_registered():
    provider = get_provider("jobspy")
    assert provider.name == "jobspy"


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        get_provider("nonexistent_provider")


def test_register_custom_provider():
    class DummyProvider(CrawlProvider):
        @property
        def name(self) -> str:
            return "dummy"

        def fetch(self, search_term, location="", results_wanted=100, **kw):
            return []

    register_provider("dummy", DummyProvider)
    p = get_provider("dummy")
    assert p.name == "dummy"
    assert p.fetch("test") == []


# ── SkillExtractor ───────────────────────────────────────────────────────────


def test_extract_skills_from_description(extractor):
    raw = _make_raw_job()
    job = extractor.extract(raw, job_id=0)
    assert "python" in job.skills
    assert "django" in job.skills
    assert "postgresql" in job.skills
    assert "docker" in job.skills
    assert "aws" in job.skills


def test_extract_seniority_senior(extractor):
    raw = _make_raw_job(title="Senior Backend Engineer")
    job = extractor.extract(raw, job_id=0)
    assert job.seniority == SeniorityLevel.SENIOR


def test_extract_seniority_junior(extractor):
    raw = _make_raw_job(title="Junior Software Developer")
    job = extractor.extract(raw, job_id=0)
    assert job.seniority == SeniorityLevel.JUNIOR


def test_extract_seniority_lead(extractor):
    raw = _make_raw_job(title="Tech Lead - Platform")
    job = extractor.extract(raw, job_id=0)
    assert job.seniority == SeniorityLevel.LEAD


def test_extract_seniority_default_mid(extractor):
    raw = _make_raw_job(
        title="Software Engineer",
        description="Build scalable web applications using Python and Django.",
    )
    job = extractor.extract(raw, job_id=0)
    assert job.seniority == SeniorityLevel.MID


def test_extract_salary_annual_to_monthly(extractor):
    raw = _make_raw_job(salary_min=60000.0, salary_max=120000.0)
    job = extractor.extract(raw, job_id=0)
    assert job.salary_min == 5000
    assert job.salary_max == 10000


def test_extract_salary_none(extractor):
    raw = _make_raw_job(salary_min=None, salary_max=None)
    job = extractor.extract(raw, job_id=0)
    assert job.salary_min == 0
    assert job.salary_max == 0


def test_extract_batch(extractor):
    raws = [_make_raw_job(source_url=f"https://example.com/{i}") for i in range(5)]
    jobs = extractor.extract_batch(raws, start_id=10)
    assert len(jobs) == 5
    assert jobs[0].job_id == 10
    assert jobs[4].job_id == 14


def test_extract_text_includes_title(extractor):
    raw = _make_raw_job(title="Senior Python Developer")
    job = extractor.extract(raw, job_id=0)
    assert "Senior Python Developer" in job.text


# ── Storage ──────────────────────────────────────────────────────────────────


def test_save_load_roundtrip():
    jobs = [_make_raw_job(source_url=f"https://example.com/{i}") for i in range(3)]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        save_raw_jobs(jobs, path)
        loaded = load_raw_jobs(path)
        assert len(loaded) == 3
        assert loaded[0].title == jobs[0].title
        assert loaded[0].source_url == jobs[0].source_url
        assert loaded[0].description == jobs[0].description
    finally:
        path.unlink(missing_ok=True)


def test_save_append():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        save_raw_jobs([_make_raw_job(source_url="a")], path)
        save_raw_jobs([_make_raw_job(source_url="b")], path)
        loaded = load_raw_jobs(path)
        assert len(loaded) == 2
    finally:
        path.unlink(missing_ok=True)


def test_load_nonexistent_returns_empty():
    assert load_raw_jobs(Path("/nonexistent/file.jsonl")) == []


def test_deduplicate():
    jobs = [
        _make_raw_job(source_url="https://example.com/1"),
        _make_raw_job(source_url="https://example.com/1"),
        _make_raw_job(source_url="https://example.com/2"),
    ]
    deduped = deduplicate(jobs)
    assert len(deduped) == 2


def test_deduplicate_keeps_no_url():
    jobs = [
        _make_raw_job(source_url=""),
        _make_raw_job(source_url=""),
    ]
    deduped = deduplicate(jobs)
    assert len(deduped) == 2  # both kept (no URL to dedup on)
