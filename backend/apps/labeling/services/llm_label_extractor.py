"""LLM-based CV-Job pair scorer."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path


def _render(template: str, **kwargs: str) -> str:
    """Replace {identifier} placeholders only — leaves JSON braces untouched."""
    return re.sub(
        r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
        lambda m: kwargs.get(m.group(1), m.group(0)),
        template,
    )

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pair_scoring.md"

SENIORITY_DISPLAY = {
    "INTERN": "Intern (0)", "JUNIOR": "Junior (1)", "MID": "Mid (2)",
    "SENIOR": "Senior (3)", "LEAD": "Lead (4)", "MANAGER": "Manager (5)",
}


@dataclass
class LabelResult:
    skill_fit:      int = 0
    seniority_fit:  int = 0
    experience_fit: int = 0
    domain_fit:     int = 0
    overall:        int = 0


def _clamp(val, lo=0, hi=2) -> int:
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return 0


def _skill_name(s) -> str:
    return s["name"] if isinstance(s, dict) else str(s)


def _fmt_skills_cv(skills: list) -> str:
    def prof(s):
        return s.get("proficiency", 3) if isinstance(s, dict) else 3
    top = sorted(skills, key=lambda s: -prof(s))[:15]
    return ", ".join(_skill_name(s) for s in top) or "(none)"


def _fmt_skills_job(skills: list) -> str:
    def imp(s):
        return s.get("importance", 3) if isinstance(s, dict) else 3
    must = [s for s in skills if imp(s) >= 4]
    nice = [s for s in skills if imp(s) < 4]
    parts = []
    if must:
        parts.append("required: " + ", ".join(_skill_name(s) for s in must[:10]))
    if nice:
        parts.append("nice-to-have: " + ", ".join(_skill_name(s) for s in nice[:5]))
    return "; ".join(parts) or "(none)"


def _fmt_experience_job(exp_min: float, exp_max: float | None) -> str:
    if not exp_min:
        return "not specified"
    if exp_max:
        return f"{exp_min:.0f}–{exp_max:.0f} years"
    return f"≥ {exp_min:.0f} years"


def extract_label(
    cv_role: str,
    cv_seniority: str,
    cv_experience: float,
    cv_education: str,
    cv_skills: list[dict],
    cv_text: str,
    job_title: str,
    job_role: str,
    job_seniority: str,
    job_experience_min: float,
    job_experience_max: float | None,
    job_skills: list[dict],
    job_text: str,
) -> LabelResult:
    from apps.llm.service import LLMService

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = _render(
        template,
        cv_role=cv_role or "other",
        cv_seniority=SENIORITY_DISPLAY.get(cv_seniority, cv_seniority),
        cv_experience=f"{cv_experience:.1f}",
        cv_education=cv_education,
        cv_skills=_fmt_skills_cv(cv_skills),
        cv_text=(cv_text or "")[:5000],
        job_title=job_title,
        job_role=job_role or "other",
        job_seniority=SENIORITY_DISPLAY.get(job_seniority, job_seniority),
        job_experience=_fmt_experience_job(job_experience_min, job_experience_max),
        job_skills=_fmt_skills_job(job_skills),
        job_description=(job_text or "")[:5000],
    )

    try:
        response = LLMService.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            feature="pair_scoring",
        )
    except Exception as exc:
        logger.warning("LLM call failed for pair scoring: %s", exc)
        raise

    # Strip <think>…</think> and code fences
    text = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        text = m.group(1).strip()

    # Extract JSON even if surrounded by extra text
    m = re.search(r"\{[\s\S]+\}", text)
    if not m:
        logger.warning("No JSON found in pair scoring response: %.200s", response)
        raise ValueError("No JSON in LLM response")

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in pair scoring: %s — %.200s", exc, response)
        raise

    return LabelResult(
        skill_fit      = _clamp(data.get("skill_fit")),
        seniority_fit  = _clamp(data.get("seniority_fit")),
        experience_fit = _clamp(data.get("experience_fit")),
        domain_fit     = _clamp(data.get("domain_fit")),
        overall        = _clamp(data.get("overall")),
    )
