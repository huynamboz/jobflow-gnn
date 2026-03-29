"""Extract structured fields from RawJob into JobData for the graph pipeline.

Uses SkillNormalizer to find canonical skills in job description text.
Infers seniority from title/description heuristics.
Computes TF-IDF skill importance when a corpus is provided.
"""

from __future__ import annotations

import math
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
    """Converts RawJob instances to JobData for the graph pipeline.

    When ``fit()`` is called with a corpus of job descriptions, skill importance
    is computed via TF-IDF: rare skills get higher importance (1-5 scale).
    Without ``fit()``, all skills default to importance=3.
    """

    def __init__(self, normalizer: SkillNormalizer) -> None:
        self._norm = normalizer
        self._idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, raws: list[RawJob]) -> SkillExtractor:
        """Compute IDF from a corpus of job descriptions.

        After fitting, ``extract()`` uses TF-IDF to assign skill importance.
        """
        n = len(raws)
        if n == 0:
            self._fitted = True
            return self

        # doc_freq: how many JDs mention each skill
        doc_freq: dict[str, int] = {}
        for raw in raws:
            skills_in_doc = set(self._extract_skills(raw.description))
            for skill in skills_in_doc:
                doc_freq[skill] = doc_freq.get(skill, 0) + 1

        # IDF: log(N / df) — common skills get low IDF, rare skills high
        self._idf = {}
        for skill, df in doc_freq.items():
            self._idf[skill] = math.log(n / df)

        self._fitted = True
        return self

    def extract(self, raw: RawJob, job_id: int) -> JobData:
        """Extract structured fields from a single RawJob."""
        skills = self._extract_skills(raw.description)
        seniority = self._infer_seniority(raw.title, raw.description)
        sal_min, sal_max = self._normalize_salary(
            raw.salary_min, raw.salary_max, raw.salary_currency
        )
        importances = self._compute_importances(skills)

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

    def _compute_importances(self, skills: list[str]) -> tuple[int, ...]:
        """Compute skill importance (1-5) using TF-IDF if fitted, else default 3."""
        if not self._fitted or not self._idf:
            return tuple(3 for _ in skills)

        if not skills:
            return ()

        # Raw IDF scores
        raw_scores = [self._idf.get(s, 0.0) for s in skills]
        min_s = min(raw_scores) if raw_scores else 0.0
        max_s = max(raw_scores) if raw_scores else 0.0

        # Normalize to 1-5
        if max_s - min_s < 1e-8:
            return tuple(3 for _ in skills)

        importances = []
        for score in raw_scores:
            normalized = (score - min_s) / (max_s - min_s)  # 0-1
            level = int(normalized * 4) + 1  # 1-5
            importances.append(min(level, 5))
        return tuple(importances)

    # ------------------------------------------------------------------
    # Skill extraction
    # ------------------------------------------------------------------

    # Single-char skills that need context to avoid false positives
    _CONTEXT_REQUIRED: dict[str, re.Pattern] = {
        "c": re.compile(r"\bC(?:\s*[/+]|(?:\s+(?:programming|language|developer|code)))", re.I),
        "r": re.compile(r"\bR(?:\s+(?:programming|language|Studio|developer|statistical))", re.I),
    }

    def _extract_skills(self, text: str) -> list[str]:
        """Find canonical skills mentioned in text.

        Strategy: split text into tokens/bigrams/trigrams,
        attempt to normalize each through the alias map.
        Deduplicate while preserving order.

        Single-char skills ("c", "r") require context patterns to avoid
        matching standalone letters in regular text.
        """
        seen: set[str] = set()
        result: list[str] = []

        # Tokenize, strip trailing punctuation
        words = [w.rstrip(".,;:") for w in re.findall(r"[\w#+.]+", text)]
        candidates: list[str] = []

        # Skip single-char tokens for unigrams (handled via context patterns)
        for w in words:
            if len(w) > 1:
                candidates.append(w)

        # Add bigrams and trigrams (these can capture "C++" "C#" etc.)
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                candidates.append(" ".join(words[i : i + n]))

        for candidate in candidates:
            canonical = self._norm.normalize(candidate)
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)

        # Check context-required skills separately
        for canonical, pattern in self._CONTEXT_REQUIRED.items():
            if canonical not in seen and pattern.search(text):
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
