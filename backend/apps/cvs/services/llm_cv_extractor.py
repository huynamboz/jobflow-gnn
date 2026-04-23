"""LLM-based CV field extractor."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "cv_extraction.md"

_EDUCATION_MAP = {
    "none": 0,
    "college": 1,
    "diploma": 1,
    "bachelor": 2,
    "undergraduate": 2,
    "master": 3,
    "masters": 3,
    "mba": 3,
    "phd": 4,
    "doctorate": 4,
}

VALID_ROLE_CATEGORIES = {
    "backend", "frontend", "fullstack", "mobile", "devops",
    "data_ml", "data_eng", "qa", "design", "ba", "other",
}


@dataclass
class CVExtractResult:
    name: str = ""
    experience_years: float = 0.0
    seniority: int = -1           # -1 = not returned by LLM; cv_service falls back to years-based rule
    role_category: str = "other"
    education: int = 2            # default BACHELOR
    skills: list[dict] = field(default_factory=list)   # [{"name": str, "proficiency": int}]
    work_experience: list[dict] = field(default_factory=list)


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def extract(raw_text: str) -> CVExtractResult:
    """Call LLM to extract structured fields from raw CV text."""
    from apps.llm.service import LLMService

    system_prompt = _load_system_prompt()

    try:
        response = LLMService.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.0,
            max_tokens=2048,
            feature="cv_extraction",
        )
    except Exception as exc:
        logger.warning("LLM call failed during CV extraction: %s", exc)
        raise

    cleaned = _strip_code_fence(response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response: %.200s", response)
        return CVExtractResult()

    education_raw = str(data.get("education", "")).lower()
    education_int = _EDUCATION_MAP.get(education_raw, 2)

    try:
        experience_years = float(data.get("experience_years") or 0)
    except (TypeError, ValueError):
        experience_years = 0.0

    # Seniority: use LLM value if present and valid; -1 signals fallback to years-based rule
    seniority_raw = data.get("seniority")
    if seniority_raw is not None:
        try:
            seniority = max(0, min(5, int(seniority_raw)))
        except (TypeError, ValueError):
            seniority = -1
    else:
        seniority = -1

    role_category = str(data.get("role_category") or "other").lower()
    if role_category not in VALID_ROLE_CATEGORIES:
        role_category = "other"

    # Skills: LLM should return canonical names; lowercase for safety
    skills = []
    for s in data.get("skills") or []:
        if isinstance(s, dict) and s.get("name"):
            skills.append({
                "name": str(s["name"]).strip().lower(),
                "proficiency": max(1, min(5, int(s.get("proficiency") or 3))),
            })

    work_experience = [
        w for w in data.get("work_experience") or []
        if isinstance(w, dict) and w.get("title")
    ]

    return CVExtractResult(
        name=str(data.get("name") or "").strip(),
        experience_years=experience_years,
        seniority=seniority,
        role_category=role_category,
        education=education_int,
        skills=skills,
        work_experience=work_experience,
    )
