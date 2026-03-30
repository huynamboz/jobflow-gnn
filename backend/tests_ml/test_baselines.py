from __future__ import annotations

import pytest

from ml_service.baselines.bm25 import BM25Scorer
from ml_service.baselines.cosine import CosineSimilarityScorer
from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.graph.schema import CVData, EducationLevel, JobData, SeniorityLevel


def _make_cv(cv_id: int = 0, skills: tuple[str, ...] = (), text: str = "") -> CVData:
    return CVData(
        cv_id=cv_id,
        seniority=SeniorityLevel.MID,
        experience_years=3.0,
        education=EducationLevel.BACHELOR,
        skills=skills,
        skill_proficiencies=tuple(3 for _ in skills),
        text=text,
    )


def _make_job(job_id: int = 0, skills: tuple[str, ...] = (), text: str = "") -> JobData:
    return JobData(
        job_id=job_id,
        seniority=SeniorityLevel.MID,
        skills=skills,
        skill_importances=tuple(3 for _ in skills),
        salary_min=2000,
        salary_max=4000,
        text=text,
    )


# ---------------------------------------------------------------------------
# CosineSimilarityScorer
# ---------------------------------------------------------------------------


class TestCosineSimilarityScorer:
    def test_score_returns_float(self, fake_embed):
        scorer = CosineSimilarityScorer(fake_embed)
        cv = _make_cv(text="python developer")
        job = _make_job(text="python engineer")
        result = scorer.score(cv, job)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_score_batch(self, fake_embed):
        scorer = CosineSimilarityScorer(fake_embed)
        cvs = [_make_cv(cv_id=i, text=f"dev {i}") for i in range(3)]
        jobs = [_make_job(job_id=i, text=f"eng {i}") for i in range(3)]
        scores = scorer.score_batch(cvs, jobs)
        assert len(scores) == 3
        for s in scores:
            assert -1.0 <= s <= 1.0

    def test_score_batch_empty(self, fake_embed):
        scorer = CosineSimilarityScorer(fake_embed)
        assert scorer.score_batch([], []) == []


# ---------------------------------------------------------------------------
# SkillOverlapScorer
# ---------------------------------------------------------------------------


class TestSkillOverlapScorer:
    def test_perfect_overlap(self):
        scorer = SkillOverlapScorer()
        cv = _make_cv(skills=("python", "java"))
        job = _make_job(skills=("python", "java"))
        assert scorer.score(cv, job) == 1.0

    def test_no_overlap(self):
        scorer = SkillOverlapScorer()
        cv = _make_cv(skills=("python",))
        job = _make_job(skills=("rust",))
        assert scorer.score(cv, job) == 0.0

    def test_partial_overlap(self):
        scorer = SkillOverlapScorer()
        cv = _make_cv(skills=("python", "java"))
        job = _make_job(skills=("python", "go"))
        # intersection=1, union=3 → 1/3
        assert scorer.score(cv, job) == pytest.approx(1 / 3)

    def test_empty_skills(self):
        scorer = SkillOverlapScorer()
        cv = _make_cv(skills=())
        job = _make_job(skills=())
        assert scorer.score(cv, job) == 0.0

    def test_score_batch_uses_default(self):
        scorer = SkillOverlapScorer()
        cvs = [_make_cv(skills=("python",)), _make_cv(skills=("java",))]
        jobs = [_make_job(skills=("python",)), _make_job(skills=("python",))]
        scores = scorer.score_batch(cvs, jobs)
        assert scores[0] == 1.0
        assert scores[1] == 0.0


# ---------------------------------------------------------------------------
# BM25Scorer
# ---------------------------------------------------------------------------


class TestBM25Scorer:
    def test_fit_and_score(self):
        cvs = [
            _make_cv(cv_id=0, text="python developer machine learning"),
            _make_cv(cv_id=1, text="java developer spring boot"),
            _make_cv(cv_id=2, text="python data science analytics"),
        ]
        scorer = BM25Scorer().fit(cvs)
        job = _make_job(text="python machine learning engineer")
        # CV 0 and CV 2 mention python; CV 0 also mentions machine learning
        s0 = scorer.score(cvs[0], job)
        s1 = scorer.score(cvs[1], job)
        s2 = scorer.score(cvs[2], job)
        assert s0 > s1  # python + ML vs java
        assert s0 > s2  # python + ML vs python only

    def test_score_without_fit_raises(self):
        scorer = BM25Scorer()
        cv = _make_cv(text="python")
        job = _make_job(text="python")
        with pytest.raises(RuntimeError, match="fit"):
            scorer.score(cv, job)

    def test_empty_corpus(self):
        scorer = BM25Scorer().fit([])
        cv = _make_cv(text="python")
        job = _make_job(text="python")
        assert scorer.score(cv, job) == 0.0

    def test_no_query_overlap(self):
        cvs = [_make_cv(text="python developer")]
        scorer = BM25Scorer().fit(cvs)
        job = _make_job(text="rust engineer")
        assert scorer.score(cvs[0], job) == 0.0
