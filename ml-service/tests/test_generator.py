import pytest

from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import SeniorityLevel


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


@pytest.fixture
def generator(normalizer):
    return SyntheticDataGenerator(normalizer, seed=42)


def test_generate_cvs_count(generator):
    cvs = generator.generate_cvs(50)
    assert len(cvs) == 50


def test_generate_cvs_deterministic(normalizer):
    g1 = SyntheticDataGenerator(normalizer, seed=42)
    g2 = SyntheticDataGenerator(normalizer, seed=42)
    cvs1 = g1.generate_cvs(10)
    cvs2 = g2.generate_cvs(10)
    for c1, c2 in zip(cvs1, cvs2):
        assert c1 == c2


def test_cv_skill_count_range(generator):
    cvs = generator.generate_cvs(100)
    for cv in cvs:
        assert 4 <= len(cv.skills) <= 12, f"CV {cv.cv_id}: {len(cv.skills)} skills"


def test_cv_proficiency_range(generator):
    cvs = generator.generate_cvs(50)
    for cv in cvs:
        for prof in cv.skill_proficiencies:
            assert 1 <= prof <= 5


def test_cv_experience_range(generator):
    cvs = generator.generate_cvs(100)
    for cv in cvs:
        assert 0 <= cv.experience_years <= 20


def test_cv_text_not_empty(generator):
    cvs = generator.generate_cvs(10)
    for cv in cvs:
        assert len(cv.text) > 0


def test_generate_jobs_count(generator):
    jobs = generator.generate_jobs(50)
    assert len(jobs) == 50


def test_job_no_intern_seniority(generator):
    jobs = generator.generate_jobs(200)
    for job in jobs:
        assert job.seniority != SeniorityLevel.INTERN


def test_job_skill_count_range(generator):
    jobs = generator.generate_jobs(100)
    for job in jobs:
        assert 3 <= len(job.skills) <= 8


def test_job_salary_range(generator):
    jobs = generator.generate_jobs(100)
    for job in jobs:
        assert job.salary_min <= job.salary_max
        assert job.salary_min >= 0
        assert job.salary_max <= 10_000


def test_job_skills_are_canonical(generator, normalizer):
    jobs = generator.generate_jobs(50)
    canonical = set(normalizer.canonical_skills)
    for job in jobs:
        for skill in job.skills:
            assert skill in canonical, f"Non-canonical skill: {skill}"
