import pytest

from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.data.skill_taxonomy import (
    SKILL_CLUSTERS,
    SKILL_SYNONYMS,
    cluster_coverage,
)


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


def test_synonyms_reference_valid_skills(normalizer):
    """All skills in SKILL_SYNONYMS must be canonical."""
    canonical = set(normalizer.canonical_skills)
    for skill in SKILL_SYNONYMS:
        assert skill in canonical, f"Synonym key '{skill}' not in canonical skills"


def test_clusters_reference_valid_skills(normalizer):
    """All skills in SKILL_CLUSTERS must be canonical."""
    canonical = set(normalizer.canonical_skills)
    for cluster_name, members in SKILL_CLUSTERS.items():
        for skill in members:
            assert skill in canonical, f"Cluster '{cluster_name}' has invalid skill '{skill}'"


def test_cluster_coverage_full():
    skills = {"react", "nodejs", "postgresql", "html_css", "javascript", "typescript"}
    cov = cluster_coverage(skills, ["fullstack_web"])
    assert cov == 1.0


def test_cluster_coverage_partial():
    skills = {"react", "nodejs"}  # 2 out of 6 in fullstack_web = 0.33 < 0.4
    cov = cluster_coverage(skills, ["fullstack_web"])
    assert cov == 0.0  # below threshold


def test_cluster_coverage_above_threshold():
    skills = {"react", "nodejs", "javascript"}  # 3/6 = 0.5 >= 0.4
    cov = cluster_coverage(skills, ["fullstack_web"])
    assert cov == 1.0  # 1 out of 1 cluster covered


def test_cluster_coverage_empty():
    assert cluster_coverage({"python"}, []) == 0.0


def test_cluster_coverage_multiple():
    # Covers fullstack_web (3/6>=0.4) and devops (3/6>=0.4)
    skills = {"react", "nodejs", "javascript", "docker", "kubernetes", "aws"}
    cov = cluster_coverage(skills, ["fullstack_web", "devops"])
    assert cov == 1.0  # both covered


def test_cluster_coverage_partial_multi():
    skills = {"react", "nodejs", "javascript"}  # fullstack_web: yes, devops: no
    cov = cluster_coverage(skills, ["fullstack_web", "devops"])
    assert cov == 0.5  # 1/2 clusters
