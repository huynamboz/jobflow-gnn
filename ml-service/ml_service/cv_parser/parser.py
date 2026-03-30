"""Parse CV from PDF/DOCX/text → CVData.

Section-based parsing:
  1. Split CV into sections (SKILLS, EXPERIENCE, PROJECTS, EDUCATION, etc.)
  2. Extract skills from SKILLS section first (highest trust)
  3. Supplement from EXPERIENCE/PROJECTS (medium trust, filter soft skills)
  4. Infer seniority, experience years, education from respective sections
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
    (re.compile(r"\b(?:engineering manager|project manager|product manager|director|head of|vp)\b", re.I), SeniorityLevel.MANAGER),
]

_EDUCATION_PATTERNS: list[tuple[re.Pattern, EducationLevel]] = [
    (re.compile(r"\b(?:ph\.?d|doctorate)\b", re.I), EducationLevel.PHD),
    (re.compile(r"\b(?:master|mba|m\.s\.|m\.tech|msc)\b", re.I), EducationLevel.MASTER),
    (re.compile(r"\b(?:bachelor|b\.s\.|b\.tech|bsc|b\.e\.?)\b", re.I), EducationLevel.BACHELOR),
    (re.compile(r"\b(?:diploma|associate|college)\b", re.I), EducationLevel.COLLEGE),
]

_YEARS_PATTERN = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience)?", re.I)

# Section header patterns (common CV section names)
_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "skills": re.compile(r"^(?:SKILLS?|TECHNICAL\s+SKILLS?|CORE\s+COMPETENC|TECHNOLOGIES|TECH\s+STACK)", re.I | re.M),
    "experience": re.compile(r"^(?:WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|PROFESSIONAL\s+EXPERIENCE)", re.I | re.M),
    "projects": re.compile(r"^(?:PROJECTS?|PERSONAL\s+PROJECTS?|SIDE\s+PROJECTS?)", re.I | re.M),
    "education": re.compile(r"^(?:EDUCATION|ACADEMIC|QUALIFICATIONS)", re.I | re.M),
    "summary": re.compile(r"^(?:SUMMARY|OBJECTIVE|ABOUT|PROFILE)", re.I | re.M),
}

# Skills that should be excluded when found only in project/experience context
# (too vague or likely false positives from descriptive text)
_CONTEXT_ONLY_SKILLS = {
    "security", "problem_solving", "communication", "teamwork",
    "leadership", "time_management", "agile",
}


class CVParser:
    """Parse CV files (PDF/DOCX/text) into CVData with section-based skill extraction."""

    def __init__(self, normalizer: SkillNormalizer) -> None:
        self._normalizer = normalizer

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
        """Parse raw CV text into CVData using section-based extraction."""
        sections = self._split_sections(text)
        skills = self._extract_skills_sectioned(sections, text)
        seniority = self._infer_seniority(text)
        experience_years = self._infer_experience_years(text)
        education = self._infer_education(text)

        proficiencies = tuple(3 for _ in skills)

        # Build embedding text: summary + experience + skills (skip volunteer/other)
        embed_parts = []
        for key in ("summary", "experience", "skills", "projects"):
            if sections.get(key):
                embed_parts.append(sections[key])
        embed_text = " ".join(embed_parts) if embed_parts else text
        words = embed_text.split()
        if len(words) > 500:
            embed_text = " ".join(words[:500])

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

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sections(text: str) -> dict[str, str]:
        """Split CV text into named sections.

        Returns dict like {"skills": "...", "experience": "...", "projects": "..."}.
        Unmatched text goes into "other".
        """
        # Find all section boundaries
        boundaries: list[tuple[int, str]] = []
        for section_name, pattern in _SECTION_PATTERNS.items():
            for match in pattern.finditer(text):
                boundaries.append((match.start(), section_name))

        if not boundaries:
            return {"other": text}

        boundaries.sort(key=lambda x: x[0])

        sections: dict[str, str] = {}
        # Text before first section → "header"
        if boundaries[0][0] > 0:
            sections["header"] = text[: boundaries[0][0]].strip()

        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()
            # Remove the header line itself
            lines = section_text.split("\n", 1)
            sections[name] = lines[1].strip() if len(lines) > 1 else ""

        return sections

    # ------------------------------------------------------------------
    # Section-based skill extraction
    # ------------------------------------------------------------------

    def _extract_skills_sectioned(
        self, sections: dict[str, str], full_text: str,
    ) -> list[str]:
        """Extract skills from full text, filtering vague skills from non-skill sections.

        Strategy: always scan full text (robust for PDF), but remove
        context-only skills (security, problem_solving, etc.) unless
        they explicitly appear in the SKILLS section.
        """
        # Step 1: Scan full text for all skills
        all_skills = self._scan_text(full_text)

        # Step 2: Find which skills are explicitly in the SKILLS section
        skill_section = sections.get("skills", "")
        trusted_skills = set(self._scan_text(skill_section)) if skill_section else set()

        # Step 3: Filter — keep context-only skills ONLY if in SKILLS section
        result: list[str] = []
        for skill in all_skills:
            if skill in _CONTEXT_ONLY_SKILLS and skill not in trusted_skills:
                continue  # vague skill only in experience/project text → skip
            result.append(skill)

        return result

    def _scan_text(self, text: str) -> list[str]:
        """Extract canonical skills from text using n-gram matching."""
        if not text:
            return []

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

        # Context-required skills (c, r)
        for canonical, pattern in SkillExtractor._CONTEXT_REQUIRED.items():
            if canonical not in seen and pattern.search(text):
                seen.add(canonical)
                result.append(canonical)

        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
