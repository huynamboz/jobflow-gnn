from __future__ import annotations

import random as _random_mod

from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.schema import (
    SENIORITY_TO_SALARY_USD,
    SENIORITY_TO_YEARS,
    CVData,
    EducationLevel,
    JobData,
    SeniorityLevel,
    SkillCategory,
)

# Seniority weights for CV generation (all levels)
_CV_SENIORITY_WEIGHTS: dict[SeniorityLevel, float] = {
    SeniorityLevel.INTERN: 0.10,
    SeniorityLevel.JUNIOR: 0.25,
    SeniorityLevel.MID: 0.30,
    SeniorityLevel.SENIOR: 0.20,
    SeniorityLevel.LEAD: 0.10,
    SeniorityLevel.MANAGER: 0.05,
}

# JD seniority weights (no intern — companies don't post intern JDs as frequently)
_JOB_SENIORITY_WEIGHTS: dict[SeniorityLevel, float] = {
    SeniorityLevel.JUNIOR: 0.25,
    SeniorityLevel.MID: 0.35,
    SeniorityLevel.SENIOR: 0.25,
    SeniorityLevel.LEAD: 0.10,
    SeniorityLevel.MANAGER: 0.05,
}

# Skill count ranges by seniority
_CV_SKILL_COUNT: dict[SeniorityLevel, tuple[int, int]] = {
    SeniorityLevel.INTERN: (4, 6),
    SeniorityLevel.JUNIOR: (5, 8),
    SeniorityLevel.MID: (6, 10),
    SeniorityLevel.SENIOR: (8, 12),
    SeniorityLevel.LEAD: (8, 12),
    SeniorityLevel.MANAGER: (7, 11),
}

# Category distribution weights by seniority (technical, soft, tool, domain)
_CATEGORY_WEIGHTS: dict[SeniorityLevel, dict[SkillCategory, float]] = {
    SeniorityLevel.INTERN: {
        SkillCategory.TECHNICAL: 0.3,
        SkillCategory.SOFT: 0.3,
        SkillCategory.TOOL: 0.3,
        SkillCategory.DOMAIN: 0.1,
    },
    SeniorityLevel.JUNIOR: {
        SkillCategory.TECHNICAL: 0.3,
        SkillCategory.SOFT: 0.2,
        SkillCategory.TOOL: 0.4,
        SkillCategory.DOMAIN: 0.1,
    },
    SeniorityLevel.MID: {
        SkillCategory.TECHNICAL: 0.25,
        SkillCategory.SOFT: 0.15,
        SkillCategory.TOOL: 0.4,
        SkillCategory.DOMAIN: 0.2,
    },
    SeniorityLevel.SENIOR: {
        SkillCategory.TECHNICAL: 0.2,
        SkillCategory.SOFT: 0.15,
        SkillCategory.TOOL: 0.35,
        SkillCategory.DOMAIN: 0.3,
    },
    SeniorityLevel.LEAD: {
        SkillCategory.TECHNICAL: 0.15,
        SkillCategory.SOFT: 0.25,
        SkillCategory.TOOL: 0.3,
        SkillCategory.DOMAIN: 0.3,
    },
    SeniorityLevel.MANAGER: {
        SkillCategory.TECHNICAL: 0.1,
        SkillCategory.SOFT: 0.35,
        SkillCategory.TOOL: 0.25,
        SkillCategory.DOMAIN: 0.3,
    },
}

# Proficiency ranges by seniority (min, max) for random.randint
_PROFICIENCY_RANGE: dict[SeniorityLevel, tuple[int, int]] = {
    SeniorityLevel.INTERN: (1, 2),
    SeniorityLevel.JUNIOR: (1, 3),
    SeniorityLevel.MID: (2, 4),
    SeniorityLevel.SENIOR: (3, 5),
    SeniorityLevel.LEAD: (3, 5),
    SeniorityLevel.MANAGER: (2, 4),
}

# Education level weights by seniority
_EDU_WEIGHTS: dict[SeniorityLevel, list[tuple[EducationLevel, float]]] = {
    SeniorityLevel.INTERN: [
        (EducationLevel.COLLEGE, 0.4),
        (EducationLevel.BACHELOR, 0.5),
        (EducationLevel.MASTER, 0.1),
    ],
    SeniorityLevel.JUNIOR: [
        (EducationLevel.COLLEGE, 0.2),
        (EducationLevel.BACHELOR, 0.6),
        (EducationLevel.MASTER, 0.2),
    ],
    SeniorityLevel.MID: [
        (EducationLevel.COLLEGE, 0.1),
        (EducationLevel.BACHELOR, 0.6),
        (EducationLevel.MASTER, 0.3),
    ],
    SeniorityLevel.SENIOR: [
        (EducationLevel.BACHELOR, 0.5),
        (EducationLevel.MASTER, 0.4),
        (EducationLevel.PHD, 0.1),
    ],
    SeniorityLevel.LEAD: [
        (EducationLevel.BACHELOR, 0.4),
        (EducationLevel.MASTER, 0.5),
        (EducationLevel.PHD, 0.1),
    ],
    SeniorityLevel.MANAGER: [
        (EducationLevel.BACHELOR, 0.3),
        (EducationLevel.MASTER, 0.5),
        (EducationLevel.PHD, 0.2),
    ],
}


class SyntheticDataGenerator:
    def __init__(self, normalizer: SkillNormalizer, seed: int = 42) -> None:
        self._norm = normalizer
        self._rng = _random_mod.Random(seed)
        self._skills_by_cat: dict[SkillCategory, list[str]] = {
            cat: normalizer.get_skills_by_category(cat) for cat in SkillCategory
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_cvs(self, n: int) -> list[CVData]:
        return [self._make_cv(i) for i in range(n)]

    def generate_jobs(self, n: int) -> list[JobData]:
        return [self._make_job(i) for i in range(n)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_choice(self, options: dict | list, *, is_list: bool = False):
        if is_list:
            items, weights = zip(*options)
        else:
            items = list(options.keys())
            weights = list(options.values())
        return self._rng.choices(items, weights=weights, k=1)[0]

    def _sample_skills(self, seniority: SeniorityLevel, count: int) -> list[str]:
        cat_weights = _CATEGORY_WEIGHTS[seniority]
        sampled: list[str] = []
        attempts = 0
        while len(sampled) < count and attempts < count * 10:
            cat = self._weighted_choice(cat_weights)
            pool = self._skills_by_cat.get(cat, [])
            if not pool:
                attempts += 1
                continue
            skill = self._rng.choice(pool)
            if skill not in sampled:
                sampled.append(skill)
            attempts += 1
        return sampled

    def _make_cv(self, idx: int) -> CVData:
        seniority = self._weighted_choice(_CV_SENIORITY_WEIGHTS)
        yr_lo, yr_hi = SENIORITY_TO_YEARS[seniority]
        experience = round(self._rng.uniform(yr_lo, yr_hi), 1)

        lo, hi = _CV_SKILL_COUNT[seniority]
        num_skills = self._rng.randint(lo, hi)
        skills = self._sample_skills(seniority, num_skills)

        prof_lo, prof_hi = _PROFICIENCY_RANGE[seniority]
        proficiencies = [self._rng.randint(prof_lo, prof_hi) for _ in skills]

        edu = self._weighted_choice(_EDU_WEIGHTS[seniority], is_list=True)

        text = self._cv_text(seniority, skills, experience, edu)

        return CVData(
            cv_id=idx,
            seniority=seniority,
            experience_years=experience,
            education=edu,
            skills=tuple(skills),
            skill_proficiencies=tuple(proficiencies),
            text=text,
        )

    def _make_job(self, idx: int) -> JobData:
        seniority = self._weighted_choice(_JOB_SENIORITY_WEIGHTS)
        num_skills = self._rng.randint(3, 8)
        skills = self._sample_skills(seniority, num_skills)
        importances = [self._rng.randint(1, 5) for _ in skills]

        sal_lo, sal_hi = SENIORITY_TO_SALARY_USD[seniority]
        salary_min = self._rng.randint(sal_lo, (sal_lo + sal_hi) // 2)
        salary_max = self._rng.randint((sal_lo + sal_hi) // 2, sal_hi)

        text = self._job_text(seniority, skills, salary_min, salary_max)

        return JobData(
            job_id=idx,
            seniority=seniority,
            skills=tuple(skills),
            skill_importances=tuple(importances),
            salary_min=salary_min,
            salary_max=salary_max,
            text=text,
        )

    # ------------------------------------------------------------------
    # Text templates (simple English for embedding)
    # ------------------------------------------------------------------

    @staticmethod
    def _cv_text(
        seniority: SeniorityLevel,
        skills: list[str],
        experience: float,
        edu: EducationLevel,
    ) -> str:
        skill_str = ", ".join(skills)
        title = seniority.name.capitalize()
        return (
            f"{title} software engineer with {experience} years of experience. "
            f"Education: {edu.name.lower()}. "
            f"Skills: {skill_str}."
        )

    @staticmethod
    def _job_text(
        seniority: SeniorityLevel,
        skills: list[str],
        salary_min: int,
        salary_max: int,
    ) -> str:
        skill_str = ", ".join(skills)
        return (
            f"Hiring {seniority.name.lower()} software engineer. "
            f"Required skills: {skill_str}. "
            f"Salary range: ${salary_min}-${salary_max} USD/month."
        )
