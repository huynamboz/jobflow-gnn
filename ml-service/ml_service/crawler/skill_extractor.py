"""Extract structured fields from RawJob into JobData for the graph pipeline.

Uses SkillNormalizer to find canonical skills in job description text.
Infers seniority from title/description heuristics.
"""

from __future__ import annotations

import re

from ml_service.crawler.base import RawJob
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import JobData, SeniorityLevel

# ---------------------------------------------------------------------------
# Seniority inference rules (applied to lowercased title + description)
# ---------------------------------------------------------------------------
_SENIORITY_PATTERNS: list[tuple[re.Pattern, SeniorityLevel]] = [
    (re.compile(r"\b(?:intern|internship|trainee)\b", re.I), SeniorityLevel.INTERN),
    (re.compile(r"\b(?:junior|jr\.?|entry.?level|associate|graduate)\b", re.I), SeniorityLevel.JUNIOR),
    (re.compile(r"\b(?:senior|sr\.?)\b", re.I), SeniorityLevel.SENIOR),
    (re.compile(r"\b(?:lead|tech.?lead|principal|staff)\b", re.I), SeniorityLevel.LEAD),
    (re.compile(r"\b(?:manager|director|head of|vp)\b", re.I), SeniorityLevel.MANAGER),
    # mid is the fallback — no explicit pattern
]

# Monthly salary normalization constant
_SALARY_NORM_ANNUAL_TO_MONTHLY = 12


class SkillExtractor:
    """Converts RawJob instances to JobData for the graph pipeline."""

    def __init__(self, normalizer: SkillNormalizer) -> None:
        self._norm = normalizer

    def extract(self, raw: RawJob, job_id: int) -> JobData:
        """Extract structured fields from a single RawJob."""
        skills = self._extract_skills(raw.description)
        seniority = self._infer_seniority(raw.title, raw.description)
        sal_min, sal_max = self._normalize_salary(
            raw.salary_min, raw.salary_max, raw.salary_currency
        )
        # Default importance = 3 for all extracted skills (no signal from raw data)
        importances = tuple(3 for _ in skills)

        return JobData(
            job_id=job_id,
            seniority=seniority,
            skills=tuple(skills),
            skill_importances=importances,
            salary_min=sal_min,
            salary_max=sal_max,
            text=f"{raw.title}. {raw.description[:2000]}",
        )

    def extract_batch(self, raws: list[RawJob], start_id: int = 0) -> list[JobData]:
        """Extract a batch of RawJobs, assigning sequential IDs."""
        return [self.extract(raw, start_id + i) for i, raw in enumerate(raws)]

    # ------------------------------------------------------------------
    # Skill extraction
    # ------------------------------------------------------------------

    def _extract_skills(self, text: str) -> list[str]:
        """Find canonical skills mentioned in text.

        Strategy: split text into tokens/bigrams/trigrams,
        attempt to normalize each through the alias map.
        Deduplicate while preserving order.
        """
        seen: set[str] = set()
        result: list[str] = []

        # Try normalizing individual words and n-grams
        # Tokenize, then strip trailing dots/commas
        words = [w.rstrip(".,;:") for w in re.findall(r"[\w#+.]+", text)]
        candidates = list(words)

        # Add bigrams and trigrams
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                candidates.append(" ".join(words[i : i + n]))

        for candidate in candidates:
            canonical = self._norm.normalize(candidate)
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)

        return result

    # ------------------------------------------------------------------
    # Seniority inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_seniority(title: str, description: str) -> SeniorityLevel:
        """Infer seniority from title first, then description. Default: MID."""
        # Check title first (more reliable signal)
        for pattern, level in _SENIORITY_PATTERNS:
            if pattern.search(title):
                return level
        # Fallback: check description
        for pattern, level in _SENIORITY_PATTERNS:
            if pattern.search(description[:500]):
                return level
        return SeniorityLevel.MID

    # ------------------------------------------------------------------
    # Salary normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_salary(
        sal_min: float | None,
        sal_max: float | None,
        currency: str,
    ) -> tuple[int, int]:
        """Normalize salary to monthly USD. Returns (0, 0) if unknown."""
        if sal_min is None and sal_max is None:
            return 0, 0

        lo = sal_min or 0.0
        hi = sal_max or lo

        # Heuristic: if values > 500, likely annual → divide by 12
        if lo > 500:
            lo = lo / _SALARY_NORM_ANNUAL_TO_MONTHLY
            hi = hi / _SALARY_NORM_ANNUAL_TO_MONTHLY

        return int(lo), int(hi)
