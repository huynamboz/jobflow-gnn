from __future__ import annotations

import pytest

from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.builder import GraphBuilder
from ml_service.training.trainer import TrainConfig, Trainer, TrainResult


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


@pytest.fixture
def tiny_dataset(normalizer, fake_embed):
    """Generate a tiny dataset for integration testing."""
    gen = SyntheticDataGenerator(normalizer, seed=42)
    cvs = gen.generate_cvs(30)
    jobs = gen.generate_jobs(50)

    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=30)
    dataset = labeler.split(pairs)

    builder = GraphBuilder(fake_embed)
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
    return data, dataset, cvs, jobs


class TestTrainer:
    def test_train_completes(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        config = TrainConfig(hidden_channels=32, num_layers=1, epochs=3, patience=5)
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)

        assert isinstance(result, TrainResult)
        assert len(result.train_losses) == 3

    def test_losses_populated(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        config = TrainConfig(hidden_channels=32, num_layers=1, epochs=5, patience=10)
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)

        assert len(result.train_losses) == 5
        for loss in result.train_losses:
            assert isinstance(loss, float)
            assert loss >= 0

    def test_val_metrics_populated(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        config = TrainConfig(hidden_channels=32, num_layers=1, epochs=3, patience=10)
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)

        assert len(result.val_metrics_history) == 3
        for metrics in result.val_metrics_history:
            assert isinstance(metrics, dict)

    def test_test_metrics_computed(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        config = TrainConfig(hidden_channels=32, num_layers=1, epochs=3, patience=10)
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)

        assert isinstance(result.test_metrics, dict)

    def test_early_stopping_triggers(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        config = TrainConfig(hidden_channels=32, num_layers=1, epochs=100, patience=2)
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)

        # Should stop well before 100 epochs
        assert len(result.train_losses) < 100

    def test_best_epoch_tracked(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        config = TrainConfig(hidden_channels=32, num_layers=1, epochs=5, patience=10)
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)

        assert 0 <= result.best_epoch < 5

    def test_custom_hybrid_weights(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        config = TrainConfig(
            hidden_channels=32,
            num_layers=1,
            epochs=2,
            patience=10,
            hybrid_alpha=0.5,
            hybrid_beta=0.3,
            hybrid_gamma=0.2,
        )
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)
        assert len(result.train_losses) == 2

    def test_default_config(self, tiny_dataset):
        data, dataset, cvs, jobs = tiny_dataset
        # Override only epochs/patience for speed
        config = TrainConfig(epochs=2, patience=5, hidden_channels=32, num_layers=1)
        trainer = Trainer(config)
        result = trainer.train(data, dataset, cvs, jobs)
        assert isinstance(result, TrainResult)
