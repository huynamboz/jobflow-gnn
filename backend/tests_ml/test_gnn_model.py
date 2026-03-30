from __future__ import annotations

import copy

import pytest
import torch

from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import EDGE_TRIPLETS, EdgeType
from ml_service.models.gnn import HeteroGraphSAGE, prepare_data_for_gnn


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


def _strip_label_edges(data):
    """Remove match/no_match edges for GNN training (prevent leakage)."""
    data = copy.copy(data)
    for etype in (EdgeType.MATCH, EdgeType.NO_MATCH):
        triplet = EDGE_TRIPLETS[etype]
        if triplet in data.edge_types:
            del data[triplet]
    return data


@pytest.fixture
def gnn_data(normalizer, fake_embed):
    """Build a small graph, strip label edges, add reverse edges."""
    gen = SyntheticDataGenerator(normalizer, seed=42)
    cvs = gen.generate_cvs(10)
    jobs = gen.generate_jobs(20)

    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=10)

    builder = GraphBuilder(fake_embed)
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
    data = _strip_label_edges(data)
    data = prepare_data_for_gnn(data)
    return data, cvs, jobs


class TestHeteroGraphSAGE:
    def test_encode_output_shapes(self, gnn_data):
        data, cvs, jobs = gnn_data
        model = HeteroGraphSAGE(metadata=data.metadata(), hidden_channels=32, num_layers=2)
        z_dict = model.encode(data)

        assert z_dict["cv"].shape == (len(cvs), 32)
        assert z_dict["job"].shape == (len(jobs), 32)
        assert "skill" in z_dict
        assert "seniority" in z_dict

    def test_decode_output_shape(self, gnn_data):
        data, _, _ = gnn_data
        model = HeteroGraphSAGE(metadata=data.metadata(), hidden_channels=32, num_layers=2)
        z_dict = model.encode(data)

        cv_idx = torch.tensor([0, 1, 2])
        job_idx = torch.tensor([0, 1, 2])
        scores = model.decode(z_dict, cv_idx, job_idx)
        assert scores.shape == (3,)

    def test_forward_output_shape(self, gnn_data):
        data, _, _ = gnn_data
        model = HeteroGraphSAGE(metadata=data.metadata(), hidden_channels=32, num_layers=2)

        cv_idx = torch.tensor([0, 1])
        job_idx = torch.tensor([0, 1])
        scores = model.forward(data, cv_idx, job_idx)
        assert scores.shape == (2,)

    def test_gradient_flow(self, gnn_data):
        data, _, _ = gnn_data
        model = HeteroGraphSAGE(metadata=data.metadata(), hidden_channels=32, num_layers=2)

        cv_idx = torch.tensor([0, 1])
        job_idx = torch.tensor([0, 1])
        scores = model.forward(data, cv_idx, job_idx)
        loss = scores.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_different_hidden_channels(self, gnn_data):
        data, cvs, _ = gnn_data
        for h in [16, 64]:
            model = HeteroGraphSAGE(metadata=data.metadata(), hidden_channels=h, num_layers=1)
            z_dict = model.encode(data)
            assert z_dict["cv"].shape[1] == h

    def test_multiple_layers(self, gnn_data):
        data, _, _ = gnn_data
        for n_layers in [1, 3]:
            model = HeteroGraphSAGE(
                metadata=data.metadata(), hidden_channels=32, num_layers=n_layers
            )
            z_dict = model.encode(data)
            assert z_dict["cv"].shape[1] == 32

    def test_encode_decode_split(self, gnn_data):
        """encode and decode should give same result as forward."""
        data, _, _ = gnn_data
        model = HeteroGraphSAGE(metadata=data.metadata(), hidden_channels=32, num_layers=2)
        model.eval()

        cv_idx = torch.tensor([0, 1])
        job_idx = torch.tensor([0, 1])

        with torch.no_grad():
            scores_forward = model.forward(data, cv_idx, job_idx)
            z_dict = model.encode(data)
            scores_split = model.decode(z_dict, cv_idx, job_idx)

        assert torch.allclose(scores_forward, scores_split)

    def test_custom_node_dims(self, gnn_data):
        data, _, _ = gnn_data
        model = HeteroGraphSAGE(
            metadata=data.metadata(),
            hidden_channels=32,
            num_layers=1,
            node_dims={"cv": 386, "job": 386, "skill": 385, "seniority": 6},
        )
        z_dict = model.encode(data)
        assert z_dict["cv"].shape[1] == 32

    def test_scores_are_finite(self, gnn_data):
        data, _, _ = gnn_data
        model = HeteroGraphSAGE(metadata=data.metadata(), hidden_channels=32, num_layers=2)

        cv_idx = torch.tensor([0, 1, 2])
        job_idx = torch.tensor([0, 1, 2])
        scores = model.forward(data, cv_idx, job_idx)
        assert torch.isfinite(scores).all()
