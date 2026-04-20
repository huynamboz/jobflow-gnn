"""Tests for per-CV full-ranking evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from ml_service.baselines.base import Scorer
from ml_service.evaluation.per_cv_evaluator import (
    PerCVEvaluator,
    PerCVResult,
    _build_cv_job_sets,
)
from ml_service.graph.schema import (
    CVData,
    DatasetSplit,
    EducationLevel,
    JobData,
    LabeledPair,
    SeniorityLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cv(cv_id: int, skills: tuple[str, ...] = ("python",)) -> CVData:
    return CVData(
        cv_id=cv_id,
        seniority=SeniorityLevel.MID,
        experience_years=3.0,
        education=EducationLevel.BACHELOR,
        skills=skills,
        skill_proficiencies=tuple(3 for _ in skills),
        text=f"CV {cv_id}",
    )


def _make_job(job_id: int, skills: tuple[str, ...] = ("python",)) -> JobData:
    return JobData(
        job_id=job_id,
        seniority=SeniorityLevel.MID,
        skills=skills,
        skill_importances=tuple(3 for _ in skills),
        salary_min=3000,
        salary_max=5000,
        text=f"Job {job_id}",
    )


class FixedScorer(Scorer):
    """Scorer that returns pre-defined scores for testing."""

    def __init__(self, score_map: dict[tuple[int, int], float], default: float = 0.0):
        self._score_map = score_map
        self._default = default

    def score(self, cv: CVData, job: JobData) -> float:
        return self._score_map.get((cv.cv_id, job.job_id), self._default)


# ---------------------------------------------------------------------------
# _build_cv_job_sets
# ---------------------------------------------------------------------------

class TestBuildCVJobSets:
    def test_basic(self):
        pairs = [
            LabeledPair(cv_id=1, job_id=10, label=1, split="test"),
            LabeledPair(cv_id=1, job_id=20, label=0, split="test"),
            LabeledPair(cv_id=2, job_id=10, label=1, split="test"),
        ]
        positives, all_jobs = _build_cv_job_sets(pairs)
        assert positives == {1: {10}, 2: {10}}
        assert all_jobs == {1: {10, 20}, 2: {10}}

    def test_empty(self):
        positives, all_jobs = _build_cv_job_sets([])
        assert positives == {}
        assert all_jobs == {}


# ---------------------------------------------------------------------------
# PerCVEvaluator
# ---------------------------------------------------------------------------

class TestPerCVEvaluator:
    """Tests for per-CV evaluation with controlled data."""

    def _make_simple_scenario(self):
        """Create a simple scenario with 2 CVs and 5 jobs.

        CV 1: test positives = {job 3, job 4}, trained on {job 0}
        CV 2: test positives = {job 2}, trained on {job 1}
        """
        cvs = [_make_cv(0, ("python", "django")), _make_cv(1, ("react", "typescript"))]
        jobs = [_make_job(j) for j in range(5)]

        dataset = DatasetSplit(
            train=[
                LabeledPair(cv_id=0, job_id=0, label=1, split="train"),
                LabeledPair(cv_id=1, job_id=1, label=1, split="train"),
            ],
            val=[],
            test=[
                LabeledPair(cv_id=0, job_id=3, label=1, split="test"),
                LabeledPair(cv_id=0, job_id=4, label=1, split="test"),
                LabeledPair(cv_id=1, job_id=2, label=1, split="test"),
            ],
        )
        return cvs, jobs, dataset

    def test_perfect_ranking(self):
        """Scorer that ranks all relevant jobs at the top."""
        cvs, jobs, dataset = self._make_simple_scenario()

        # CV 0: job3=0.9, job4=0.8 are top (both relevant). Candidates: jobs 1,2,3,4
        # CV 1: job2=0.9 is top (relevant). Candidates: jobs 0,2,3,4
        scorer = FixedScorer({
            (0, 1): 0.1, (0, 2): 0.2, (0, 3): 0.9, (0, 4): 0.8,
            (1, 0): 0.1, (1, 2): 0.9, (1, 3): 0.2, (1, 4): 0.1,
        })

        evaluator = PerCVEvaluator(cvs, jobs, dataset)
        result = evaluator.evaluate(scorer, ks=(2, 5))

        assert result.num_cvs_evaluated == 2
        # CV 0: 2 relevant in top-2 of 4 candidates → recall@2 = 2/2 = 1.0
        # CV 1: 1 relevant in top-2 of 4 candidates → recall@2 = 1/1 = 1.0
        assert result.avg_metrics["recall@2"] == pytest.approx(1.0)
        assert result.avg_metrics["ndcg@2"] == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Scorer that ranks relevant jobs at the bottom."""
        cvs, jobs, dataset = self._make_simple_scenario()

        # CV 0: relevant jobs get lowest scores
        # CV 1: relevant job gets lowest score
        scorer = FixedScorer({
            (0, 1): 0.9, (0, 2): 0.8, (0, 3): 0.1, (0, 4): 0.05,
            (1, 0): 0.9, (1, 2): 0.05, (1, 3): 0.8, (1, 4): 0.7,
        })

        evaluator = PerCVEvaluator(cvs, jobs, dataset)
        result = evaluator.evaluate(scorer, ks=(2,))

        # Both CVs have 0 relevant in top-2
        assert result.avg_metrics["recall@2"] == pytest.approx(0.0)
        assert result.avg_metrics["precision@2"] == pytest.approx(0.0)

    def test_excludes_training_jobs(self):
        """Training jobs should not appear in candidate ranking."""
        cvs = [_make_cv(0, ("python",))]
        jobs = [_make_job(j) for j in range(3)]

        dataset = DatasetSplit(
            train=[LabeledPair(cv_id=0, job_id=0, label=1, split="train")],
            val=[LabeledPair(cv_id=0, job_id=1, label=0, split="val")],
            test=[LabeledPair(cv_id=0, job_id=2, label=1, split="test")],
        )

        # Job 0 has highest score, but should be excluded (training)
        # Job 1 should also be excluded (val)
        # Only job 2 remains as candidate
        scorer = FixedScorer({(0, 0): 0.99, (0, 1): 0.8, (0, 2): 0.5})

        evaluator = PerCVEvaluator(cvs, jobs, dataset)
        result = evaluator.evaluate(scorer, ks=(1,))

        assert result.num_cvs_evaluated == 1
        # Only job 2 as candidate, it's relevant → perfect
        assert result.avg_metrics["recall@1"] == pytest.approx(1.0)

    def test_min_test_positives_filter(self):
        """CVs with fewer than min_test_positives should be skipped."""
        cvs = [_make_cv(0), _make_cv(1)]
        jobs = [_make_job(j) for j in range(5)]

        dataset = DatasetSplit(
            train=[],
            val=[],
            test=[
                LabeledPair(cv_id=0, job_id=0, label=1, split="test"),  # only 1 positive
                LabeledPair(cv_id=1, job_id=1, label=1, split="test"),
                LabeledPair(cv_id=1, job_id=2, label=1, split="test"),  # 2 positives
            ],
        )

        scorer = FixedScorer({}, default=0.5)
        evaluator = PerCVEvaluator(cvs, jobs, dataset, min_test_positives=2)
        result = evaluator.evaluate(scorer, ks=(5,))

        # Only CV 1 has >= 2 test positives
        assert result.num_cvs_evaluated == 1
        assert result.num_cvs_with_test_positives == 2

    def test_no_eligible_cvs(self):
        """Should return empty result if no CVs have enough test positives."""
        cvs = [_make_cv(0)]
        jobs = [_make_job(0)]
        dataset = DatasetSplit(train=[], val=[], test=[])

        scorer = FixedScorer({})
        evaluator = PerCVEvaluator(cvs, jobs, dataset)
        result = evaluator.evaluate(scorer, ks=(5,))

        assert result.num_cvs_evaluated == 0
        assert result.avg_metrics == {}

    def test_evaluate_with_score_fn(self):
        """Test the custom score_fn interface."""
        cvs, jobs, dataset = self._make_simple_scenario()

        def custom_scorer(cv: CVData, job: JobData) -> float:
            # CV 0: relevant jobs (3, 4) get high scores
            if cv.cv_id == 0 and job.job_id in (3, 4):
                return 0.9
            # CV 1: relevant job (2) gets high score
            if cv.cv_id == 1 and job.job_id == 2:
                return 0.9
            return 0.1

        evaluator = PerCVEvaluator(cvs, jobs, dataset)
        result = evaluator.evaluate_with_score_fn(custom_scorer, ks=(2,))

        assert result.num_cvs_evaluated == 2
        assert result.avg_metrics["recall@2"] == pytest.approx(1.0)

    def test_best_worst_cvs_populated(self):
        """Best and worst CV lists should be populated."""
        cvs, jobs, dataset = self._make_simple_scenario()
        scorer = FixedScorer({
            (0, 1): 0.1, (0, 2): 0.2, (0, 3): 0.9, (0, 4): 0.8,
            (1, 0): 0.1, (1, 2): 0.9, (1, 3): 0.2, (1, 4): 0.1,
        })

        evaluator = PerCVEvaluator(cvs, jobs, dataset)
        result = evaluator.evaluate(scorer, ks=(5, 10))

        assert len(result.best_cvs) > 0
        assert len(result.worst_cvs) > 0

    def test_per_cv_metrics_breakdown(self):
        """Each CV should have its own metrics in the breakdown."""
        cvs, jobs, dataset = self._make_simple_scenario()
        scorer = FixedScorer({}, default=0.5)

        evaluator = PerCVEvaluator(cvs, jobs, dataset)
        result = evaluator.evaluate(scorer, ks=(2,))

        assert 0 in result.per_cv_metrics
        assert 1 in result.per_cv_metrics
        assert "recall@2" in result.per_cv_metrics[0]
        assert "mrr" in result.per_cv_metrics[0]
