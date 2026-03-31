"""
JobFlow-GNN — Crawl job postings and convert to graph-ready data.

Usage:
    cd backend
    python run_crawl.py

Outputs:
    data/raw_jobs.jsonl  — raw crawled data (append-safe)
    data/jobs.jsonl      — extracted JobData (graph-ready)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ml_service.crawler import get_provider
from ml_service.data.skill_extractor import SkillExtractor
from ml_service.crawler.storage import deduplicate, load_raw_jobs, save_raw_jobs
from ml_service.data.skill_normalization import SkillNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("crawl")

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw_jobs.jsonl"
SKILL_ALIAS_PATH = "ml_service/data/skill-alias.json"

# Search queries — IT/Software focused
QUERIES = [
    ("software engineer", ""),
    ("python developer", ""),
    ("frontend developer", ""),
    ("backend developer", ""),
    ("fullstack developer", ""),
    ("data engineer", ""),
    ("devops engineer", ""),
    ("machine learning engineer", ""),
]

RESULTS_PER_QUERY = 50  # JobSpy results per search


def main() -> None:
    t_start = time.time()

    # ─── Step 1: Crawl ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Step 1: Crawl job postings via JobSpy")
    logger.info("=" * 60)

    provider = get_provider("jobspy", sites=["indeed", "glassdoor", "linkedin"])
    all_raw = []

    for term, location in QUERIES:
        try:
            jobs = provider.fetch(
                search_term=term,
                location=location,
                results_wanted=RESULTS_PER_QUERY,
            )
            all_raw.extend(jobs)
            logger.info("  %s → %d jobs", term, len(jobs))
        except Exception as e:
            logger.error("  %s → FAILED: %s", term, e)

    # Deduplicate
    before = len(all_raw)
    all_raw = deduplicate(all_raw)
    logger.info("Total: %d raw → %d after dedup", before, len(all_raw))

    # Save raw
    written = save_raw_jobs(all_raw, RAW_PATH)
    logger.info("Saved %d raw jobs to %s", written, RAW_PATH)

    # ─── Step 2: Extract skills + convert to JobData ────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Step 2: Extract skills & convert to JobData")
    logger.info("=" * 60)

    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    extractor = SkillExtractor(normalizer)

    # Load all raw jobs (including previous runs)
    all_stored = load_raw_jobs(RAW_PATH)
    all_stored = deduplicate(all_stored)
    logger.info("Loaded %d unique raw jobs from %s", len(all_stored), RAW_PATH)

    job_datas = extractor.extract_batch(all_stored)

    # Filter out jobs with 0 skills
    jobs_with_skills = [j for j in job_datas if len(j.skills) > 0]
    jobs_no_skills = len(job_datas) - len(jobs_with_skills)
    logger.info("Extracted: %d jobs with skills, %d skipped (no skills)", len(jobs_with_skills), jobs_no_skills)

    # Stats
    all_skills = set()
    for j in jobs_with_skills:
        all_skills.update(j.skills)
    avg_skills = sum(len(j.skills) for j in jobs_with_skills) / max(len(jobs_with_skills), 1)
    logger.info("Unique skills found: %d", len(all_skills))
    logger.info("Avg skills per job: %.1f", avg_skills)

    # Seniority distribution
    seniority_dist: dict[str, int] = {}
    for j in jobs_with_skills:
        level = j.seniority.name
        seniority_dist[level] = seniority_dist.get(level, 0) + 1
    logger.info("Seniority distribution: %s", json.dumps(seniority_dist, indent=2))

    # Top skills
    skill_freq: dict[str, int] = {}
    for j in jobs_with_skills:
        for s in j.skills:
            skill_freq[s] = skill_freq.get(s, 0) + 1
    top_20 = sorted(skill_freq.items(), key=lambda x: -x[1])[:20]
    logger.info("Top 20 skills:")
    for skill, count in top_20:
        logger.info("  %-25s %d", skill, count)

    total_time = time.time() - t_start
    logger.info("")
    logger.info("Done in %.1fs. %d graph-ready jobs.", total_time, len(jobs_with_skills))


if __name__ == "__main__":
    main()
