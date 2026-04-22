"""Matching service — singleton wrapper around ml_service InferenceEngine.

Loads model once, reuses across requests.
"""

from __future__ import annotations

import logging
import threading

from django.conf import settings

logger = logging.getLogger(__name__)

_engine = None
_parser = None
_lock = threading.Lock()


def _get_engine():
    """Lazy-load InferenceEngine singleton."""
    global _engine
    if _engine is not None:
        return _engine

    with _lock:
        if _engine is not None:
            return _engine

        from ml_service.data.skill_normalization import SkillNormalizer
        from ml_service.embedding import get_provider
        from ml_service.inference import InferenceEngine

        logger.info("Loading ML engine from %s...", settings.ML_CHECKPOINT_DIR)
        normalizer = SkillNormalizer(settings.ML_SKILL_ALIAS_PATH)
        provider = get_provider()

        _engine = InferenceEngine.from_checkpoint(
            settings.ML_CHECKPOINT_DIR,
            normalizer=normalizer,
            embedding_provider=provider,
        )
        logger.info("ML engine ready: %d CVs, %d jobs", _engine.num_cvs, _engine.num_jobs)

        # Override checkpoint job skills with DB canonical skills (more accurate)
        try:
            _refresh_job_skills_from_db(_engine)
        except Exception as e:
            logger.warning("Job skill refresh skipped: %s", e)

        return _engine


def _refresh_job_skills_from_db(engine) -> None:
    """Replace checkpoint job skills with DB-stored canonical skills.

    The checkpoint's SkillExtractor can miss or mis-extract skills from job
    descriptions. The DB stores canonical_name skills (via JobSkill M2M) that
    were extracted and normalised when the job was crawled, giving more accurate
    skill data for scoring without needing to retrain.
    """
    from apps.jobs.models import JobSkill
    from ml_service.graph.schema import JobData

    job_ids = [j.job_id for j in engine.job_pool]
    if not job_ids:
        return

    # One query: all skill entries for all jobs in pool
    skill_rows = (
        JobSkill.objects
        .filter(job_id__in=job_ids)
        .select_related("skill")
        .values_list("job_id", "skill__canonical_name", "importance")
    )

    db_skills: dict[int, list[tuple[str, int]]] = {}
    for job_id, canonical, importance in skill_rows:
        db_skills.setdefault(job_id, []).append((canonical, importance))

    if not db_skills:
        logger.info("DB has no skills for engine job pool — keeping checkpoint skills.")
        return

    updated = 0
    new_jobs = []
    for job in engine.job_pool:
        entries = db_skills.get(job.job_id)
        if entries:
            new_jobs.append(JobData(
                job_id=job.job_id,
                seniority=job.seniority,
                skills=tuple(s for s, _ in entries),
                skill_importances=tuple(i for _, i in entries),
                salary_min=job.salary_min,
                salary_max=job.salary_max,
                text=job.text,
            ))
            updated += 1
        else:
            new_jobs.append(job)

    engine.replace_job_skills(new_jobs)
    logger.info("Refreshed DB skills for %d/%d jobs.", updated, len(new_jobs))


def _get_parser():
    """Lazy-load CVParser singleton."""
    global _parser
    if _parser is not None:
        return _parser

    with _lock:
        if _parser is not None:
            return _parser

        from ml_service.cv_parser import CVParser
        from ml_service.data.skill_normalization import SkillNormalizer

        normalizer = SkillNormalizer(settings.ML_SKILL_ALIAS_PATH)
        _parser = CVParser(normalizer)
        return _parser


_SOFT_SKILLS = frozenset({
    "communication", "teamwork", "leadership", "problem_solving",
    "time_management", "agile", "security",
})


def _clean_title(raw: str) -> str:
    """Return first non-empty line of title (strips LinkedIn card metadata)."""
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return raw.strip()


def _filter_soft_skills(skills: list[str]) -> list[str]:
    return [s for s in skills if s not in _SOFT_SKILLS]


def _enrich(results) -> list[dict]:
    """Enrich raw match results with DB job fields."""
    from apps.jobs.models import Job

    job_ids = [r.job_id for r in results]
    db_jobs = {
        j.id: j
        for j in Job.objects.filter(id__in=job_ids).select_related("company", "platform")
    }
    enriched = []
    for r in results:
        j = db_jobs.get(r.job_id)
        raw_title = j.title if j else r.title
        enriched.append({
            "job_id": r.job_id,
            "score": r.score,
            "eligible": r.eligible,
            "matched_skills": _filter_soft_skills(list(r.matched_skills)),
            "missing_skills": _filter_soft_skills(list(r.missing_skills)),
            "seniority_match": r.seniority_match,
            "title": _clean_title(raw_title),
            "company_name": j.company.name if j and j.company else "",
            "location": j.location if j else "",
            "job_type": j.job_type if j else "",
            "salary_min": j.salary_min if j else 0,
            "salary_max": j.salary_max if j else 0,
            "source_url": j.source_url if j else "",
        })
    return enriched


def match_cv_text(cv_text: str, top_k: int = 10) -> list[dict]:
    """Match CV text against all jobs. Returns list of match results."""
    engine = _get_engine()
    results = engine.match_cv_text(cv_text, top_k=top_k)
    return _enrich(results)


def match_cv_file(file_path: str, top_k: int = 10) -> list[dict]:
    """Parse CV file (PDF/DOCX) and match against all jobs."""
    parser = _get_parser()
    cv_data = parser.parse_file(file_path)
    if not cv_data.skills:
        return []

    engine = _get_engine()
    results = engine.match_cv(cv_data, top_k=top_k)
    return _enrich(results)


def parse_cv_file(file_path: str) -> dict:
    """Parse CV file and return structured data (debug)."""
    parser = _get_parser()
    cv = parser.parse_file(file_path)
    return {
        "seniority": cv.seniority.name,
        "experience_years": cv.experience_years,
        "education": cv.education.name,
        "skills": list(cv.skills),
    }


def parse_cv_text(cv_text: str) -> dict:
    """Parse CV text and return structured data (debug)."""
    parser = _get_parser()
    cv = parser.parse_text(cv_text)
    return {
        "seniority": cv.seniority.name,
        "experience_years": cv.experience_years,
        "education": cv.education.name,
        "skills": list(cv.skills),
    }
