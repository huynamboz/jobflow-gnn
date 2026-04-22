"""LLM-based JD (Job Description) field extractor."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "jd_extraction.md"

VALID_JOB_TYPES = {"full-time", "part-time", "contract", "remote", "hybrid", "on-site"}
VALID_SALARY_TYPES = {"hourly", "monthly", "annual", "unknown"}


@dataclass
class JDExtractResult:
    title: str = ""
    company: str = ""
    location: str = ""
    seniority: int = 2
    job_type: str = "full-time"
    salary_min: int = 0
    salary_max: int = 0
    salary_currency: str = "USD"
    salary_type: str = "unknown"          # "hourly" | "monthly" | "annual" | "unknown"
    experience_min: float = 0.0           # years
    experience_max: Optional[float] = None
    degree_requirement: int = 0           # 0–5
    skills: list[dict] = field(default_factory=list)  # [{"name": str, "importance": int}]


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val or default)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def extract(raw_text: str) -> JDExtractResult:
    """Call LLM to extract structured fields from raw job description text."""
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
            feature="jd_extraction",
        )
    except Exception as exc:
        logger.warning("LLM call failed during JD extraction: %s", exc)
        return JDExtractResult()

    cleaned = _strip_code_fence(response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response: %.200s", response)
        return JDExtractResult()

    seniority = max(0, min(5, _safe_int(data.get("seniority"), 2)))
    degree_requirement = max(0, min(5, _safe_int(data.get("degree_requirement"), 0)))

    salary_min = _safe_int(data.get("salary_min"), 0)
    salary_max = _safe_int(data.get("salary_max"), 0)

    salary_type = str(data.get("salary_type") or "").lower()
    if salary_type not in VALID_SALARY_TYPES:
        salary_type = "monthly" if salary_min > 0 else "unknown"

    job_type = str(data.get("job_type") or "full-time").lower()
    if job_type not in VALID_JOB_TYPES:
        job_type = "full-time"

    experience_min = _safe_float(data.get("experience_min"), 0.0)
    exp_max_raw = data.get("experience_max")
    experience_max = _safe_float(exp_max_raw) if exp_max_raw is not None else None

    skills = [
        s for s in (data.get("skills") or [])
        if isinstance(s, dict) and s.get("name")
    ]

    return JDExtractResult(
        title=str(data.get("title") or "").strip(),
        company=str(data.get("company") or "").strip(),
        location=str(data.get("location") or "").strip(),
        seniority=seniority,
        job_type=job_type,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=str(data.get("salary_currency") or "USD").upper(),
        salary_type=salary_type,
        experience_min=experience_min,
        experience_max=experience_max,
        degree_requirement=degree_requirement,
        skills=skills,
    )
