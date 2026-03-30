from ml_service.graph.schema import (
    EDGE_TRIPLETS,
    SENIORITY_TO_SALARY_USD,
    SENIORITY_TO_YEARS,
    CVData,
    DatasetSplit,
    EdgeType,
    EducationLevel,
    JobData,
    LabeledPair,
    NodeType,
    SeniorityLevel,
    SkillCategory,
)


def test_node_types():
    assert NodeType.CV == "cv"
    assert NodeType.JOB == "job"
    assert NodeType.SKILL == "skill"
    assert NodeType.SENIORITY == "seniority"
    assert len(NodeType) == 4


def test_edge_types():
    assert len(EdgeType) == 6
    assert EdgeType.HAS_SKILL == "has_skill"
    assert EdgeType.MATCH == "match"


def test_edge_triplets():
    assert len(EDGE_TRIPLETS) == 6
    assert EDGE_TRIPLETS[EdgeType.HAS_SKILL] == ("cv", "has_skill", "skill")
    assert EDGE_TRIPLETS[EdgeType.MATCH] == ("cv", "match", "job")


def test_seniority_level():
    assert SeniorityLevel.INTERN == 0
    assert SeniorityLevel.MANAGER == 5
    assert len(SeniorityLevel) == 6


def test_skill_category():
    assert SkillCategory.TECHNICAL == 0
    assert SkillCategory.DOMAIN == 3
    assert len(SkillCategory) == 4


def test_education_level():
    assert EducationLevel.NONE == 0
    assert EducationLevel.PHD == 4
    assert len(EducationLevel) == 5


def test_seniority_mappings():
    assert len(SENIORITY_TO_YEARS) == 6
    assert len(SENIORITY_TO_SALARY_USD) == 6
    for level in SeniorityLevel:
        lo, hi = SENIORITY_TO_YEARS[level]
        assert lo <= hi
        sal_lo, sal_hi = SENIORITY_TO_SALARY_USD[level]
        assert sal_lo <= sal_hi


def test_cv_data_frozen():
    cv = CVData(
        cv_id=0,
        seniority=SeniorityLevel.MID,
        experience_years=3.0,
        education=EducationLevel.BACHELOR,
        skills=("python", "react"),
        skill_proficiencies=(4, 3),
        text="test",
    )
    assert cv.cv_id == 0
    assert cv.seniority == SeniorityLevel.MID


def test_job_data_frozen():
    job = JobData(
        job_id=0,
        seniority=SeniorityLevel.SENIOR,
        skills=("python",),
        skill_importances=(5,),
        salary_min=3500,
        salary_max=6000,
        text="test",
    )
    assert job.job_id == 0


def test_labeled_pair_defaults():
    pair = LabeledPair(cv_id=0, job_id=1, label=1)
    assert pair.split == "train"


def test_dataset_split():
    ds = DatasetSplit()
    assert ds.train == []
    assert ds.val == []
    assert ds.test == []
