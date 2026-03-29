"""Parse CV from PDF/DOCX/text → CVData.

Extracts text, skills, seniority, experience years, education.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ml_service.crawler.skill_extractor import SkillExtractor
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import CVData, EducationLevel, SeniorityLevel

logger = logging.getLogger(__name__)

_SENIORITY_PATTERNS: list[tuple[re.Pattern, SeniorityLevel]] = [
    (re.compile(r"\b(?:intern|internship|trainee)\b", re.I), SeniorityLevel.INTERN),
    (re.compile(r"\b(?:junior|jr\.?|entry.?level|graduate)\b", re.I), SeniorityLevel.JUNIOR),
    (re.compile(r"\b(?:senior|sr\.?)\b", re.I), SeniorityLevel.SENIOR),
    (re.compile(r"\b(?:lead|tech.?lead|principal|staff)\b", re.I), SeniorityLevel.LEAD),
    (re.compile(r"\b(?:manager|director|head of|vp)\b", re.I), SeniorityLevel.MANAGER),
]

_EDUCATION_PATTERNS: list[tuple[re.Pattern, EducationLevel]] = [
    (re.compile(r"\b(?:ph\.?d|doctorate)\b", re.I), EducationLevel.PHD),
    (re.compile(r"\b(?:master|mba|m\.s\.|m\.tech|msc)\b", re.I), EducationLevel.MASTER),
    (re.compile(r"\b(?:bachelor|b\.s\.|b\.tech|bsc|b\.e\.?)\b", re.I), EducationLevel.BACHELOR),
    (re.compile(r"\b(?:diploma|associate|college)\b", re.I), EducationLevel.COLLEGE),
]

_YEARS_PATTERN = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience)?", re.I)


class CVParser:
    """Parse CV files (PDF/DOCX/text) into CVData."""

    def __init__(self, normalizer: SkillNormalizer) -> None:
        self._normalizer = normalizer
        self._extractor = SkillExtractor(normalizer)

    def parse_file(self, path: str | Path, cv_id: int = 0) -> CVData:
        """Parse a CV file (PDF or DOCX) into CVData."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            text = self._extract_pdf(path)
        elif suffix in (".docx", ".doc"):
            text = self._extract_docx(path)
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")

        return self.parse_text(text, cv_id=cv_id)

    def parse_text(self, text: str, cv_id: int = 0) -> CVData:
        """Parse raw CV text into CVData."""
        skills = self._extract_skills(text)
        seniority = self._infer_seniority(text)
        experience_years = self._infer_experience_years(text)
        education = self._infer_education(text)

        proficiencies = tuple(3 for _ in skills)

        # Truncate text for embedding (keep first 500 words)
        words = text.split()
        embed_text = " ".join(words[:500]) if len(words) > 500 else text

        logger.info(
            "Parsed CV: %d skills, seniority=%s, exp=%.1fy, edu=%s",
            len(skills), seniority.name, experience_years, education.name,
        )

        return CVData(
            cv_id=cv_id,
            seniority=seniority,
            experience_years=experience_years,
            education=education,
            skills=tuple(skills),
            skill_proficiencies=proficiencies,
            text=embed_text,
        )

    def _extract_skills(self, text: str) -> list[str]:
        """Extract canonical skills from text using n-gram matching."""
        seen: set[str] = set()
        result: list[str] = []

        words = [w.rstrip(".,;:") for w in re.findall(r"[\w#+.]+", text)]
        candidates: list[str] = [w for w in words if len(w) > 1]

        for n in (2, 3):
            for i in range(len(words) - n + 1):
                candidates.append(" ".join(words[i: i + n]))

        for candidate in candidates:
            canonical = self._normalizer.normalize(candidate)
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)

        # Context-required skills
        for canonical, pattern in SkillExtractor._CONTEXT_REQUIRED.items():
            if canonical not in seen and pattern.search(text):
                seen.add(canonical)
                result.append(canonical)

        return result

    @staticmethod
    def _infer_seniority(text: str) -> SeniorityLevel:
        for pattern, level in _SENIORITY_PATTERNS:
            if pattern.search(text[:1000]):
                return level
        return SeniorityLevel.MID

    @staticmethod
    def _infer_experience_years(text: str) -> float:
        matches = _YEARS_PATTERN.findall(text)
        if matches:
            return float(max(int(m) for m in matches))
        return 0.0

    @staticmethod
    def _infer_education(text: str) -> EducationLevel:
        for pattern, level in _EDUCATION_PATTERNS:
            if pattern.search(text):
                return level
        return EducationLevel.BACHELOR

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        import pdfplumber

        text_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    @staticmethod
    def _extract_docx(path: Path) -> str:
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
