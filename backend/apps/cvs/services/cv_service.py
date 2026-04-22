"""CV service: parse + save to DB."""

from __future__ import annotations

import logging
from pathlib import Path

from django.db import transaction

from apps.cvs.models import CV, CVSkill
from apps.skills.services import SkillService

logger = logging.getLogger(__name__)


class CVService:
    """Parse CV files and save structured data to DB."""

    def __init__(self):
        from ml_service.cv_parser import CVParser
        from ml_service.data.skill_normalization import SkillNormalizer

        self._normalizer = SkillNormalizer()
        self._parser = CVParser(self._normalizer)
        self._skill_service = SkillService()

    def save_from_upload_with_llm(self, file_path: str, user=None) -> CV:
        """Parse PDF → LLM extract → save to DB. Falls back to rule-based if LLM unavailable."""
        raw_text = self._extract_raw_text(file_path)

        from apps.cvs.services.llm_cv_extractor import extract as llm_extract
        result = llm_extract(raw_text)

        # If LLM returned nothing useful, fall back to rule-based parser
        if not result.skills and not result.work_experience:
            logger.info("LLM extraction empty, falling back to rule-based parser")
            return self.save_from_file(file_path, user=user, source="upload")

        # Infer seniority from experience_years
        seniority = self._years_to_seniority(result.experience_years)

        # Normalize skills via SkillNormalizer then resolve to DB records
        normalized_skills: list[tuple[str, int]] = []
        for s in result.skills:
            canonical = self._normalizer.normalize(s["name"])
            if canonical:
                proficiency = max(1, min(5, int(s.get("proficiency") or 3)))
                normalized_skills.append((canonical, proficiency))

        with transaction.atomic():
            cv = CV.objects.create(
                user=user,
                file_name=Path(file_path).name,
                raw_text=raw_text,
                parsed_text=raw_text,
                candidate_name=result.name,
                seniority=seniority,
                experience_years=result.experience_years,
                education=result.education,
                work_experience=result.work_experience,
                source="upload",
            )
            for skill_name, proficiency in normalized_skills:
                skill = self._skill_service.get_or_create(skill_name)
                if skill:
                    CVSkill.objects.get_or_create(
                        cv=cv, skill=skill,
                        defaults={"proficiency": proficiency},
                    )

        logger.info(
            "LLM CV #%d: name=%r exp=%.1f edu=%d skills=%d",
            cv.id, result.name, result.experience_years, result.education, len(normalized_skills),
        )
        return cv

    @staticmethod
    def _extract_raw_text(file_path: str) -> str:
        """Extract raw text from PDF/DOCX/TXT."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            import pdfplumber
            pages = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n".join(pages)
        if suffix == ".docx":
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        # .txt fallback
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _years_to_seniority(years: float) -> int:
        if years < 1:
            return 0   # INTERN
        if years < 3:
            return 1   # JUNIOR
        if years < 6:
            return 2   # MID
        if years < 9:
            return 3   # SENIOR
        return 4       # LEAD

    def save_from_extracted(self, data: dict, user=None) -> CV:
        """Save CV from pre-extracted (and user-edited) JSON data."""
        skills_data = data.get("skills") or []
        with transaction.atomic():
            cv = CV.objects.create(
                user=user,
                file_name=data.get("file_name", ""),
                raw_text=data.get("raw_text", ""),
                parsed_text=data.get("raw_text", ""),
                candidate_name=data.get("candidate_name", ""),
                seniority=int(data.get("seniority", 2)),
                experience_years=float(data.get("experience_years", 0)),
                education=int(data.get("education", 2)),
                work_experience=data.get("work_experience") or [],
                source="upload",
            )
            for skill in skills_data:
                canonical = self._normalizer.normalize(skill.get("name", ""))
                if canonical:
                    skill_obj = self._skill_service.get_or_create(canonical)
                    if skill_obj:
                        CVSkill.objects.get_or_create(
                            cv=cv, skill=skill_obj,
                            defaults={"proficiency": max(1, min(5, int(skill.get("proficiency") or 3)))},
                        )
        logger.info("Saved extracted CV #%d: %d skills", cv.id, len(skills_data))
        return cv

    def save_from_file(self, file_path: str, user=None, source: str = "upload") -> CV:
        """Parse CV file and save to DB."""
        cv_data = self._parser.parse_file(file_path)
        return self._save_cv(cv_data, file_path=file_path, user=user, source=source)

    def save_from_text(self, text: str, user=None, source: str = "upload") -> CV:
        """Parse CV text and save to DB."""
        cv_data = self._parser.parse_text(text)
        return self._save_cv(cv_data, raw_text=text, user=user, source=source)

    def _save_cv(self, cv_data, file_path=None, raw_text=None, user=None, source="upload", source_category="") -> CV:
        """Save parsed CV data to DB."""
        with transaction.atomic():
            cv = CV.objects.create(
                user=user,
                file_name=Path(file_path).name if file_path else "",
                raw_text=raw_text or cv_data.text,
                parsed_text=cv_data.text,
                seniority=cv_data.seniority,
                experience_years=cv_data.experience_years,
                education=cv_data.education,
                source=source,
                source_category=source_category,
            )

            for skill_name, proficiency in zip(cv_data.skills, cv_data.skill_proficiencies):
                skill = self._skill_service.get_or_create(skill_name)
                if skill:
                    CVSkill.objects.get_or_create(
                        cv=cv, skill=skill,
                        defaults={"proficiency": proficiency},
                    )

        logger.info("Saved CV #%d: %d skills, seniority=%s", cv.id, len(cv_data.skills), cv_data.seniority.name)
        return cv

    @staticmethod
    def to_cv_data(cv: CV):
        """Convert Django CV model → ml_service CVData."""
        from ml_service.graph.schema import CVData, EducationLevel, SeniorityLevel

        skills = tuple(cv.cv_skills.values_list("skill__canonical_name", flat=True))
        proficiencies = tuple(cv.cv_skills.values_list("proficiency", flat=True))

        return CVData(
            cv_id=cv.id,
            seniority=SeniorityLevel(cv.seniority),
            experience_years=cv.experience_years,
            education=EducationLevel(cv.education),
            skills=skills,
            skill_proficiencies=proficiencies,
            text=cv.parsed_text or cv.raw_text,
        )

    @staticmethod
    def get_all_cv_data() -> list:
        """Query all active CVs and convert to CVData list."""
        from ml_service.graph.schema import CVData, EducationLevel, SeniorityLevel

        cvs = CV.objects.filter(is_active=True).prefetch_related("cv_skills__skill")
        result = []
        for cv in cvs:
            skills = tuple(cs.skill.canonical_name for cs in cv.cv_skills.all())
            proficiencies = tuple(cs.proficiency for cs in cv.cv_skills.all())
            if len(skills) >= 2:
                result.append(CVData(
                    cv_id=cv.id,
                    seniority=SeniorityLevel(cv.seniority),
                    experience_years=cv.experience_years,
                    education=EducationLevel(cv.education),
                    skills=skills,
                    skill_proficiencies=proficiencies,
                    text=cv.parsed_text or cv.raw_text,
                ))
        return result
