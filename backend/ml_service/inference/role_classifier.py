"""Infer role category from skills/text for role-based matching penalty.

Roles: frontend, backend, fullstack, devops, data, mobile, security, other.
If CV role != JD role → penalty on matching score.
"""

from __future__ import annotations

import re

# Role → defining skills (if >= 2 match, assign this role)
_ROLE_SKILLS: dict[str, set[str]] = {
    "frontend": {"react", "vuejs", "angular", "nextjs", "html_css", "tailwind", "sass", "bootstrap", "redux", "typescript", "javascript", "css", "html", "vite", "webpack"},
    "backend": {"django", "fastapi", "flask", "spring", "express", "nestjs", "nodejs", "laravel", "rails", "postgresql", "mysql", "mongodb", "redis", "kafka", "rabbitmq", "grpc", "graphql"},
    "devops": {"docker", "kubernetes", "terraform", "ansible", "ci_cd", "aws", "gcp", "azure", "helm", "argocd", "prometheus", "grafana", "jenkins", "nginx", "linux"},
    "data": {"spark", "airflow", "hadoop", "hive", "flink", "snowflake", "databricks", "dbt", "bigquery", "pandas", "numpy", "data_engineering", "data_science", "etl"},
    "ml": {"machine_learning", "deep_learning", "pytorch", "tensorflow", "scikit_learn", "nlp", "computer_vision", "llm", "xgboost", "mlflow", "sagemaker", "langchain"},
    "mobile": {"react_native", "flutter", "ios", "android", "swift", "kotlin"},
    "security": {"security", "cybersecurity"},
}

# Role → title keywords
_ROLE_TITLE_PATTERNS: dict[str, re.Pattern] = {
    "frontend": re.compile(r"\b(?:front.?end|ui|ux|react|vue|angular)\b", re.I),
    "backend": re.compile(r"\b(?:back.?end|server|api|django|spring|node)\b", re.I),
    "fullstack": re.compile(r"\b(?:full.?stack)\b", re.I),
    "devops": re.compile(r"\b(?:devops|sre|platform|infra|cloud|reliability)\b", re.I),
    "data": re.compile(r"\b(?:data\s+(?:engineer|analyst|scientist)|etl|bi\b|analytics)\b", re.I),
    "ml": re.compile(r"\b(?:machine\s+learning|ml\b|ai\b|deep\s+learning|nlp|computer\s+vision)\b", re.I),
    "mobile": re.compile(r"\b(?:mobile|ios|android|react\s+native|flutter)\b", re.I),
    "security": re.compile(r"\b(?:security|cyber|penetration|soc)\b", re.I),
}

# Roles that are compatible (no penalty between them)
_COMPATIBLE_ROLES: dict[str, set[str]] = {
    "frontend": {"frontend", "fullstack"},
    "backend": {"backend", "fullstack"},
    "fullstack": {"frontend", "backend", "fullstack"},
    "devops": {"devops", "backend"},
    "data": {"data", "ml", "backend"},
    "ml": {"ml", "data"},
    "mobile": {"mobile", "frontend"},
    "security": {"security", "devops", "backend"},
    "other": set(),  # compatible with everything
}


def infer_role(skills: tuple[str, ...] | set[str], text: str = "") -> str:
    """Infer primary role from skills and/or text.

    Returns one of: frontend, backend, fullstack, devops, data, ml, mobile, security, other.
    """
    skill_set = set(skills)

    # Check title patterns first (strongest signal)
    for role, pattern in _ROLE_TITLE_PATTERNS.items():
        if pattern.search(text[:200]):  # check first 200 chars (title area)
            return role

    # Count skill matches per role
    scores: dict[str, int] = {}
    for role, defining_skills in _ROLE_SKILLS.items():
        overlap = len(skill_set & defining_skills)
        if overlap >= 2:
            scores[role] = overlap

    if not scores:
        return "other"

    # If both frontend and backend skills → fullstack
    if "frontend" in scores and "backend" in scores:
        return "fullstack"

    return max(scores, key=lambda r: scores[r])


def role_match_penalty(cv_role: str, job_role: str) -> float:
    """Penalty multiplier for role mismatch.

    Returns:
        1.0  — same role or compatible → no penalty
        0.7  — partial mismatch (e.g., frontend vs data)
        0.5  — strong mismatch (e.g., frontend vs security)
    """
    if cv_role == "other" or job_role == "other":
        return 1.0  # can't determine → no penalty

    if cv_role == job_role:
        return 1.0

    compatible = _COMPATIBLE_ROLES.get(cv_role, set())
    if job_role in compatible:
        return 1.0

    # Partial compatibility check (reverse)
    reverse_compatible = _COMPATIBLE_ROLES.get(job_role, set())
    if cv_role in reverse_compatible:
        return 1.0

    return 0.7
