from __future__ import annotations

import numpy as np
import pytest

from ml_service.evaluation.metrics import (
    auc_roc,
    compute_all_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert recall_at_k(y_true, y_scores, k=2) == 1.0

    def test_partial_ranking(self):
        y_true = np.array([1, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert recall_at_k(y_true, y_scores, k=2) == 0.5

    def test_zero_recall(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.05, 0.01])
        assert recall_at_k(y_true, y_scores, k=2) == 0.0

    def test_no_positives(self):
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.5, 0.3, 0.1])
        assert recall_at_k(y_true, y_scores, k=2) == 0.0

    def test_k_larger_than_n(self):
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.9, 0.5, 0.8])
        assert recall_at_k(y_true, y_scores, k=10) == 1.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_is_relevant(self):
        y_true = np.array([1, 0, 0])
        y_scores = np.array([0.9, 0.5, 0.1])
        assert mrr(y_true, y_scores) == 1.0

    def test_second_is_relevant(self):
        y_true = np.array([0, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.1])
        assert mrr(y_true, y_scores) == 0.5

    def test_no_relevant(self):
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.9, 0.5, 0.1])
        assert mrr(y_true, y_scores) == 0.0

    def test_third_is_relevant(self):
        y_true = np.array([0, 0, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.1])
        assert mrr(y_true, y_scores) == pytest.approx(1.0 / 3)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCGAtK:
    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert ndcg_at_k(y_true, y_scores, k=2) == pytest.approx(1.0)

    def test_worst_ranking(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert ndcg_at_k(y_true, y_scores, k=2) == pytest.approx(0.0)

    def test_no_positives(self):
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.9, 0.5, 0.1])
        assert ndcg_at_k(y_true, y_scores, k=2) == 0.0

    def test_partial_ranking(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.5, 0.9, 0.8, 0.1])
        # Ranked by score: [1](0), [2](1), [0](1), [3](0)
        # top-2 gains: 0, 1 → DCG = 0/log2(2) + 1/log2(3) = 0.6309
        # ideal: 1, 1 → IDCG = 1/log2(2) + 1/log2(3) = 1.6309
        expected = (0.0 + 1.0 / np.log2(3)) / (1.0 + 1.0 / np.log2(3))
        assert ndcg_at_k(y_true, y_scores, k=2) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# auc_roc
# ---------------------------------------------------------------------------


class TestAUCROC:
    def test_perfect_separation(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert auc_roc(y_true, y_scores) == 1.0

    def test_single_class(self):
        y_true = np.array([1, 1, 1])
        y_scores = np.array([0.9, 0.5, 0.1])
        assert auc_roc(y_true, y_scores) == 0.0

    def test_random(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_scores = rng.rand(100)
        result = auc_roc(y_true, y_scores)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        y_true = np.array([1, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        result = compute_all_metrics(y_true, y_scores, ks=(5, 10))
        assert "recall@5" in result
        assert "recall@10" in result
        assert "ndcg@5" in result
        assert "ndcg@10" in result
        assert "mrr" in result
        assert "auc_roc" in result

    def test_custom_ks(self):
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.1])
        result = compute_all_metrics(y_true, y_scores, ks=(2, 3))
        assert "recall@2" in result
        assert "recall@3" in result
