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
