"""Tests for inference module (checkpoint + engine)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import JobData
from ml_service.inference.checkpoint import load_checkpoint, save_checkpoint
from ml_service.inference.engine import InferenceEngine, MatchResult
from ml_service.models.gnn import HeteroGraphSAGE, prepare_data_for_gnn
from ml_service.training.trainer import Trainer, TrainConfig, _strip_label_edges
from tests_ml.conftest import FakeEmbeddingProvider


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


@pytest.fixture
def trained_artifacts(normalizer):
    """Build a small trained model + graph + data for testing."""
    fake_embed = FakeEmbeddingProvider(dim=384, seed=42)
    gen = SyntheticDataGenerator(normalizer, seed=42)
    cvs = gen.generate_cvs(20)
    jobs = gen.generate_jobs(30)

    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=30)
    dataset = labeler.split(pairs)

    builder = GraphBuilder(fake_embed)
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)

    config = TrainConfig(hidden_channels=32, num_layers=1, epochs=3, patience=3, lr=1e-3)
    trainer = Trainer(config)
    result = trainer.train(data, dataset, cvs, jobs)

    # Rebuild model to get a saveable reference
    data_clean = _strip_label_edges(data)
    data_prepared = prepare_data_for_gnn(data_clean)
    model = HeteroGraphSAGE(
        metadata=data_prepared.metadata(),
        hidden_channels=32,
        num_layers=1,
    )
    model.eval()

    return model, data, cvs, jobs, normalizer, fake_embed


# ── Checkpoint ───────────────────────────────────────────────────────────────


def test_save_load_checkpoint(trained_artifacts):
    model, data, cvs, jobs, normalizer, _ = trained_artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt"
        save_checkpoint(path, model, data, cvs, jobs, metadata={"test": True})

        model2, data2, cvs2, jobs2, meta = load_checkpoint(path, hidden_channels=32, num_layers=1)

        assert len(cvs2) == len(cvs)
        assert len(jobs2) == len(jobs)
        assert cvs2[0].cv_id == cvs[0].cv_id
        assert jobs2[0].job_id == jobs[0].job_id
        assert meta["test"] is True
        assert data2["cv"].x.shape == data["cv"].x.shape


def test_checkpoint_files_created(trained_artifacts):
    model, data, cvs, jobs, _, _ = trained_artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt"
        save_checkpoint(path, model, data, cvs, jobs)
        assert (path / "model.pt").exists()
        assert (path / "graph.pt").exists()
        assert (path / "cvs.json").exists()
        assert (path / "jobs.json").exists()
        assert (path / "metadata.json").exists()


# ── MatchResult ──────────────────────────────────────────────────────────────


def test_match_result_frozen():
    r = MatchResult(cv_id=1, score=0.8, eligible=True, matched_skills=("python",), missing_skills=(), seniority_match=True)
    with pytest.raises(AttributeError):
        r.score = 0.5


# ── InferenceEngine ──────────────────────────────────────────────────────────


def test_engine_match_returns_results(trained_artifacts):
    model, data, cvs, jobs, normalizer, fake_embed = trained_artifacts
    engine = InferenceEngine(
        model=model, data=data, cvs=cvs, jobs=jobs,
        normalizer=normalizer, embedding_provider=fake_embed,
    )
    results = engine.match("Python developer with Django and PostgreSQL experience", top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r, MatchResult) for r in results)


def test_engine_match_sorted_descending(trained_artifacts):
    model, data, cvs, jobs, normalizer, fake_embed = trained_artifacts
    engine = InferenceEngine(
        model=model, data=data, cvs=cvs, jobs=jobs,
        normalizer=normalizer, embedding_provider=fake_embed,
    )
    results = engine.match("Software engineer Python React AWS", top_k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_engine_match_has_skill_info(trained_artifacts):
    model, data, cvs, jobs, normalizer, fake_embed = trained_artifacts
    engine = InferenceEngine(
        model=model, data=data, cvs=cvs, jobs=jobs,
        normalizer=normalizer, embedding_provider=fake_embed,
    )
    results = engine.match("Looking for Python and React developer", top_k=3)
    for r in results:
        assert isinstance(r.matched_skills, tuple)
        assert isinstance(r.missing_skills, tuple)
        assert isinstance(r.seniority_match, bool)


def test_engine_num_cvs(trained_artifacts):
    model, data, cvs, jobs, normalizer, fake_embed = trained_artifacts
    engine = InferenceEngine(
        model=model, data=data, cvs=cvs, jobs=jobs,
        normalizer=normalizer, embedding_provider=fake_embed,
    )
    assert engine.num_cvs == len(cvs)


def test_engine_match_job_data(trained_artifacts):
    model, data, cvs, jobs, normalizer, fake_embed = trained_artifacts
    engine = InferenceEngine(
        model=model, data=data, cvs=cvs, jobs=jobs,
        normalizer=normalizer, embedding_provider=fake_embed,
    )
    results = engine.match_job_data(jobs[0], top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r, MatchResult) for r in results)


def test_engine_empty_query(trained_artifacts):
    model, data, cvs, jobs, normalizer, fake_embed = trained_artifacts
    engine = InferenceEngine(
        model=model, data=data, cvs=cvs, jobs=jobs,
        normalizer=normalizer, embedding_provider=fake_embed,
    )
    results = engine.match("no skills mentioned here at all xyz abc", top_k=5)
    # May return empty if no skills extracted
    assert isinstance(results, list)
