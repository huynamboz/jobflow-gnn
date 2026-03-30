import pytest

from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import SkillCategory


@pytest.fixture
def normalizer(skill_alias_path):
    return SkillNormalizer(path=skill_alias_path)


def test_canonical_skill_count(normalizer):
    assert len(normalizer.canonical_skills) == 143


def test_normalize_exact(normalizer):
    assert normalizer.normalize("python") == "python"
    assert normalizer.normalize("javascript") == "javascript"


def test_normalize_alias(normalizer):
    assert normalizer.normalize("ReactJS") == "react"
    assert normalizer.normalize("Node.js") == "nodejs"
    assert normalizer.normalize("PostgreSQL") == "postgresql"
    assert normalizer.normalize("C++") == "cpp"
    assert normalizer.normalize("C#") == "csharp"


def test_normalize_case_insensitive(normalizer):
    assert normalizer.normalize("PYTHON") == "python"
    assert normalizer.normalize("Python3") == "python"
    assert normalizer.normalize("tensorflow") == "tensorflow"


def test_normalize_unknown(normalizer):
    assert normalizer.normalize("nonexistent_skill_xyz") is None


def test_skill_catalog(normalizer):
    catalog = normalizer.skill_catalog
    assert len(catalog) == 143
    assert catalog["python"] == SkillCategory.TECHNICAL
    assert catalog["react"] == SkillCategory.TOOL
    assert catalog["communication"] == SkillCategory.SOFT
    assert catalog["machine_learning"] == SkillCategory.DOMAIN


def test_get_skills_by_category(normalizer):
    technical = normalizer.get_skills_by_category(SkillCategory.TECHNICAL)
    assert "python" in technical
    assert "javascript" in technical
    # Tools should NOT be in technical
    assert "react" not in technical

    soft = normalizer.get_skills_by_category(SkillCategory.SOFT)
    assert "communication" in soft
    assert "teamwork" in soft
