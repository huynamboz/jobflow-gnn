"""Job service: RawJob → DB (Platform + Company + Job + Skills)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.db import transaction

from apps.jobs.models import Job, JobSkill, Platform
from apps.jobs.services.platform_service import PlatformService
from apps.skills.services import SkillService

if TYPE_CHECKING:
    from ml_service.crawler.base import RawJob

logger = logging.getLogger(__name__)

# Map crawl source names to platform info
_PLATFORM_MAP = {
    "indeed": {"name": "Indeed", "base_url": "https://indeed.com"},
    "linkedin": {"name": "LinkedIn", "base_url": "https://linkedin.com"},
    "adzuna": {"name": "Adzuna", "base_url": "https://adzuna.com"},
    "remotive": {"name": "Remotive", "base_url": "https://remotive.com"},
}


class JobService:
    """Save crawled jobs to DB with auto-create Platform/Company/Skills."""

    def __init__(self):
        from ml_service.crawler.storage import compute_fingerprint
        from ml_service.data.skill_extractor import SkillExtractor
        from ml_service.data.skill_normalization import SkillNormalizer

        self._normalizer = SkillNormalizer()
        self._extractor = SkillExtractor(self._normalizer)
        self._compute_fingerprint = compute_fingerprint
        self._skill_service = SkillService()

    def save_raw_job(self, raw: "RawJob") -> Job | None:
        """Save a single RawJob to DB. Returns Job or None if duplicate."""
        # Get or create platform
        source = raw.source or "unknown"
        platform_info = _PLATFORM_MAP.get(source, {"name": source.title(), "base_url": ""})
        platform = PlatformService.get_or_create_platform(**platform_info)

        # Compute fingerprint
        fingerprint = self._compute_fingerprint(raw)

        # Check duplicate per platform
        if Job.objects.filter(platform=platform, fingerprint=fingerprint).exists():
            return None

        # Get or create company
        company = PlatformService.get_or_create_company(
            name=raw.company,
            platform=platform,
            logo_url=getattr(raw, "company_logo_url", ""),
            profile_url=getattr(raw, "company_url", ""),
        )

        # Extract skills + seniority
        job_data = self._extractor.extract(raw, job_id=0)

        # Infer job_type
        job_type = getattr(raw, "job_type", "") or ""
        if job_type:
            # Take first valid choice
            for choice in Job.JobType.values:
                if choice in job_type.lower():
                    job_type = choice
                    break
            else:
                job_type = Job.JobType.OTHER

        # Create job
        with transaction.atomic():
            job = Job.objects.create(
                platform=platform,
                company=company,
                title=raw.title,
                description=raw.description[:10000],
                location=raw.location,
                seniority=job_data.seniority,
                job_type=job_type,
                salary_min=job_data.salary_min,
                salary_max=job_data.salary_max,
                salary_currency=raw.salary_currency,
                source_url=raw.source_url,
                fingerprint=fingerprint,
                applicant_count=getattr(raw, "applicant_count", ""),
                date_posted=raw.date_posted,
            )

            # Create skill associations
            for skill_name, importance in zip(job_data.skills, job_data.skill_importances):
                skill = self._skill_service.get_or_create(skill_name)
                if skill:
                    JobSkill.objects.get_or_create(
                        job=job, skill=skill,
                        defaults={"importance": importance},
                    )

        return job

    def save_raw_jobs_batch(self, raws: list["RawJob"]) -> dict:
        """Save multiple RawJobs. Returns stats."""
        created = 0
        skipped = 0
        failed = 0

        for raw in raws:
            try:
                job = self.save_raw_job(raw)
                if job:
                    created += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error("Failed to save job '%s': %s", raw.title[:50], e)
                failed += 1

        logger.info("Saved %d jobs (skipped %d duplicates, %d failed)", created, skipped, failed)
        return {"created": created, "skipped": skipped, "failed": failed}

    @staticmethod
    def to_job_data(job: Job):
        """Convert Django Job model → ml_service JobData."""
        from ml_service.graph.schema import JobData

        skills = tuple(job.job_skills.values_list("skill__canonical_name", flat=True))
        importances = tuple(job.job_skills.values_list("importance", flat=True))

        return JobData(
            job_id=job.id,
            seniority=job.seniority,
            skills=skills,
            skill_importances=importances,
            salary_min=job.salary_min,
            salary_max=job.salary_max,
            text=f"{job.title}. {job.description[:2000]}",
        )

    @staticmethod
    def get_all_job_data() -> list:
        """Query all active jobs and convert to JobData list."""
        from ml_service.graph.schema import JobData

        jobs = Job.objects.filter(is_active=True).prefetch_related("job_skills__skill")
        result = []
        for job in jobs:
            skills = tuple(js.skill.canonical_name for js in job.job_skills.all())
            importances = tuple(js.importance for js in job.job_skills.all())
            if len(skills) >= 2:
                result.append(JobData(
                    job_id=job.id,
                    seniority=job.seniority,
                    skills=skills,
                    skill_importances=importances,
                    salary_min=job.salary_min,
                    salary_max=job.salary_max,
                    text=f"{job.title}. {job.description[:2000]}",
                ))
        return result
