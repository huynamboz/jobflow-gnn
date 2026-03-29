import pytest

from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import CVData, EducationLevel, JobData, SeniorityLevel


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


@pytest.fixture
def small_data(normalizer):
    gen = SyntheticDataGenerator(normalizer, seed=42)
    cvs = gen.generate_cvs(50)
    jobs = gen.generate_jobs(80)
    return cvs, jobs


def test_skill_overlap():
    cv = CVData(
        cv_id=0,
        seniority=SeniorityLevel.MID,
        experience_years=3.0,
        education=EducationLevel.BACHELOR,
        skills=("python", "react", "nodejs", "git"),
        skill_proficiencies=(4, 3, 3, 2),
        text="test",
    )
    job = JobData(
        job_id=0,
        seniority=SeniorityLevel.MID,
        skills=("python", "react", "postgresql", "docker"),
        skill_importances=(5, 4, 4, 3),
        salary_min=2000,
        salary_max=4000,
        text="test",
    )
    overlap = PairLabeler._skill_overlap(cv, job)
    assert overlap == 0.5  # 2 out of 4


def test_seniority_distance():
    cv = CVData(
        cv_id=0,
        seniority=SeniorityLevel.JUNIOR,
        experience_years=1.0,
        education=EducationLevel.BACHELOR,
        skills=("python",),
        skill_proficiencies=(3,),
        text="test",
    )
    job = JobData(
        job_id=0,
        seniority=SeniorityLevel.SENIOR,
        skills=("python",),
        skill_importances=(5,),
        salary_min=3500,
        salary_max=6000,
        text="test",
    )
    dist = PairLabeler._seniority_distance(cv, job)
    assert dist == 2  # |1 - 3| = 2


def test_create_pairs_has_both_labels(small_data):
    cvs, jobs = small_data
    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=50)
    labels = {p.label for p in pairs}
    assert 1 in labels
    assert 0 in labels


def test_create_pairs_positive_ratio(small_data):
    cvs, jobs = small_data
    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=50)
    n_pos = sum(1 for p in pairs if p.label == 1)
    n_neg = sum(1 for p in pairs if p.label == 0)
    # Ratio should be ~1:3 (pos : neg), allow some flex from sampling constraints
    assert n_pos > 0
    assert n_neg >= n_pos


def test_split_no_leak(small_data):
    cvs, jobs = small_data
    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=50)
    ds = labeler.split(pairs)

    train_ids = {(p.cv_id, p.job_id) for p in ds.train}
    val_ids = {(p.cv_id, p.job_id) for p in ds.val}
    test_ids = {(p.cv_id, p.job_id) for p in ds.test}

    # No pair should appear in multiple splits
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_split_proportions(small_data):
    cvs, jobs = small_data
    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=100)
    ds = labeler.split(pairs)

    total = len(ds.train) + len(ds.val) + len(ds.test)
    assert total == len(pairs)
    # Roughly 75/15/10
    assert len(ds.train) / total > 0.65
    assert len(ds.val) / total > 0.05
    assert len(ds.test) / total > 0.05


def test_split_assigns_correct_labels(small_data):
    cvs, jobs = small_data
    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=50)
    ds = labeler.split(pairs)

    for p in ds.train:
        assert p.split == "train"
    for p in ds.val:
        assert p.split == "val"
    for p in ds.test:
        assert p.split == "test"
