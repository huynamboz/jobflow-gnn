import numpy as np
import pytest

from ml_service.embedding.base import EmbeddingProvider
from ml_service.embedding.factory import get_provider
from ml_service.embedding.multilingual import MultilingualProvider

from .conftest import FakeEmbeddingProvider


def test_fake_provider_shape(fake_embed):
    result = fake_embed.encode(["hello", "world"])
    assert result.shape == (2, 384)


def test_fake_provider_normalized(fake_embed):
    result = fake_embed.encode(["test"])
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5


def test_fake_provider_is_embedding_provider(fake_embed):
    assert isinstance(fake_embed, EmbeddingProvider)


def test_multilingual_raises():
    provider = MultilingualProvider()
    with pytest.raises(NotImplementedError):
        provider.encode(["hello"])
    with pytest.raises(NotImplementedError):
        _ = provider.dim


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_provider("nonexistent")


def test_fake_provider_deterministic():
    p1 = FakeEmbeddingProvider(seed=42)
    p2 = FakeEmbeddingProvider(seed=42)
    r1 = p1.encode(["hello"])
    r2 = p2.encode(["hello"])
    np.testing.assert_array_equal(r1, r2)
