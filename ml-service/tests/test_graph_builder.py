import pytest
import torch

from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.builder import GraphBuilder


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


@pytest.fixture
def built_graph(normalizer, fake_embed):
    gen = SyntheticDataGenerator(normalizer, seed=42)
    cvs = gen.generate_cvs(20)
    jobs = gen.generate_jobs(30)

    labeler = PairLabeler(seed=42)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=20)

    builder = GraphBuilder(fake_embed)
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
    return data, cvs, jobs, pairs


def test_cv_node_shape(built_graph):
    data, cvs, _, _ = built_graph
    assert data["cv"].x.shape[0] == len(cvs)
    assert data["cv"].x.shape[1] == 386  # 384 embedding + exp_years + edu_level


def test_job_node_shape(built_graph):
    data, _, jobs, _ = built_graph
    assert data["job"].x.shape[0] == len(jobs)
    assert data["job"].x.shape[1] == 386  # 384 embedding + salary_min + salary_max


def test_skill_node_shape(built_graph, normalizer):
    data, _, _, _ = built_graph
    n_skills = len(normalizer.canonical_skills)
    assert data["skill"].x.shape[0] == n_skills
    assert data["skill"].x.shape[1] == 385  # 384 embedding + category


def test_seniority_node_shape(built_graph):
    data, _, _, _ = built_graph
    assert data["seniority"].x.shape == (6, 6)
    # Should be identity matrix
    assert torch.equal(data["seniority"].x, torch.eye(6))


def test_has_skill_edges(built_graph):
    data, _, _, _ = built_graph
    edge_index = data["cv", "has_skill", "skill"].edge_index
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    edge_attr = data["cv", "has_skill", "skill"].edge_attr
    assert edge_attr.shape[0] == edge_index.shape[1]
    # Proficiency in [1, 5]
    assert edge_attr.min() >= 1
    assert edge_attr.max() <= 5


def test_requires_skill_edges(built_graph):
    data, _, _, _ = built_graph
    edge_index = data["job", "requires_skill", "skill"].edge_index
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    edge_attr = data["job", "requires_skill", "skill"].edge_attr
    assert edge_attr.min() >= 1
    assert edge_attr.max() <= 5


def test_seniority_edges(built_graph):
    data, cvs, jobs, _ = built_graph
    hs_edge = data["cv", "has_seniority", "seniority"].edge_index
    assert hs_edge.shape == (2, len(cvs))
    # All seniority indices should be in [0, 5]
    assert hs_edge[1].max() <= 5
    assert hs_edge[1].min() >= 0

    rs_edge = data["job", "requires_seniority", "seniority"].edge_index
    assert rs_edge.shape == (2, len(jobs))


def test_match_nomatch_edges(built_graph):
    data, _, _, pairs = built_graph
    match_edges = data["cv", "match", "job"].edge_index
    nomatch_edges = data["cv", "no_match", "job"].edge_index

    n_pos = sum(1 for p in pairs if p.label == 1)
    n_neg = sum(1 for p in pairs if p.label == 0)

    assert match_edges.shape[1] == n_pos
    assert nomatch_edges.shape[1] == n_neg
