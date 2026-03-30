"""
Train the Stage 2 XGBoost reranker on labeled pairs.

Uses the same labeled data as GNN training but trains a separate
XGBoost model on rich feature vectors for final ranking.

Usage:
    cd backend
    python run_train_save.py         # train GNN first (if not done)
    python run_train_reranker.py     # train reranker on top
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ml_service.crawler.resume_loader import load_resumes
from ml_service.crawler.skill_extractor import SkillExtractor
from ml_service.crawler.storage import deduplicate, load_raw_jobs
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.graph.schema import JobData
from ml_service.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_reranker")

RAW_JOBS_PATH = Path("data/raw_jobs.jsonl")
SKILL_ALIAS_PATH = "ml_service/data/skill-alias.json"
CHECKPOINT_DIR = "checkpoints/latest"
NUM_POSITIVE_PAIRS = 1500
SEED = 42


def main() -> None:
    t_start = time.time()

    # Load engine from checkpoint
    logger.info("Loading engine from checkpoint...")
    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    provider = get_provider()

    engine = InferenceEngine.from_checkpoint(
        CHECKPOINT_DIR,
        normalizer=normalizer,
        embedding_provider=provider,
    )
    logger.info("Engine loaded: %d CVs, %d jobs", engine.num_cvs, engine.num_jobs)

    # Load real JDs for labeling
    logger.info("Loading real JDs for labeling...")
    raw_jobs = deduplicate(load_raw_jobs(RAW_JOBS_PATH))
    extractor = SkillExtractor(normalizer)
    extractor.fit(raw_jobs)
    all_jobs = extractor.extract_batch(raw_jobs)
    jobs = [j for j in all_jobs if len(j.skills) >= 2]
    jobs = [
        JobData(job_id=i, seniority=j.seniority, skills=j.skills,
                skill_importances=j.skill_importances,
                salary_min=j.salary_min, salary_max=j.salary_max, text=j.text)
        for i, j in enumerate(jobs)
    ]

    # Use engine's CV pool
    cvs = engine.cv_pool

    # Create labeled pairs
    logger.info("Creating labeled pairs...")
    labeler = PairLabeler(seed=SEED)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=NUM_POSITIVE_PAIRS, use_skill_relations=True)

    cv_id_to_idx = {cv.cv_id: i for i, cv in enumerate(cvs)}
    job_id_to_idx = {job.job_id: i for i, job in enumerate(jobs)}

    cv_indices = []
    job_indices = []
    labels = []
    for p in pairs:
        ci = cv_id_to_idx.get(p.cv_id)
        ji = job_id_to_idx.get(p.job_id)
        if ci is not None and ji is not None:
            cv_indices.append(ci)
            job_indices.append(ji)
            labels.append(p.label)

    logger.info("Training reranker on %d pairs (pos=%d, neg=%d)...",
                len(labels), sum(labels), len(labels) - sum(labels))

    # Train
    metrics = engine.train_reranker(cv_indices, job_indices, labels)
    logger.info("Reranker metrics: %s", metrics)

    # Feature importance
    importance = engine.reranker.feature_importance()
    logger.info("Feature importance:")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        logger.info("  %-30s %.4f", name, imp)

    # --- Calibration: fit Platt scaling on validation scores ---
    logger.info("Fitting score calibration...")
    val_scores = []
    val_labels = []
    # Use a subset for calibration (score each pair with stage1)
    from ml_service.data.labeler import PairLabeler as _PL
    cal_labeler = _PL(seed=99)
    cal_pairs = cal_labeler.create_pairs(cvs, jobs, num_positive=500, use_skill_relations=True)
    # Balance: equal positive and negative for calibration
    pos_pairs = [p for p in cal_pairs if p.label == 1]
    neg_pairs = [p for p in cal_pairs if p.label == 0]
    n_cal = min(len(pos_pairs), len(neg_pairs))
    balanced_pairs = pos_pairs[:n_cal] + neg_pairs[:n_cal]
    for p in balanced_pairs:
        ci = cv_id_to_idx.get(p.cv_id)
        ji = job_id_to_idx.get(p.job_id)
        if ci is not None and ji is not None:
            score = engine._score_pair(cvs[ci], jobs[ji])
            val_scores.append(score)
            val_labels.append(p.label)

    engine.calibrate(val_scores, val_labels)
    logger.info("Calibration fitted on %d samples", len(val_scores))

    total = time.time() - t_start
    logger.info("Done in %.1fs", total)


if __name__ == "__main__":
    main()
