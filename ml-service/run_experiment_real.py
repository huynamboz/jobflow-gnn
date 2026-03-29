"""
JobFlow-GNN — Experiment with REAL job data + Synthetic CVs

Loads crawled JDs from data/raw_jobs.jsonl, generates synthetic CVs
with skill distributions drawn from real JD skill frequencies,
then trains GNN and benchmarks against baselines.

Usage:
    cd ml-service
    python run_experiment_real.py
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ml_service.baselines.bm25 import BM25Scorer
from ml_service.baselines.cosine import CosineSimilarityScorer
from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.crawler.skill_extractor import SkillExtractor
from ml_service.crawler.storage import deduplicate, load_raw_jobs
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.evaluation.metrics import compute_all_metrics
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import (
    SENIORITY_TO_YEARS,
    CVData,
    EducationLevel,
    JobData,
    SeniorityLevel,
)
from ml_service.training.trainer import Trainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("experiment_real")

# ─── Config ───────────────────────────────────────────────────────────────────

RAW_JOBS_PATH = Path("data/raw_jobs.jsonl")
SKILL_ALIAS_PATH = "../roadmap/week1/skill-alias.json"

NUM_CVS = 200
NUM_POSITIVE_PAIRS = 600
NOISE_RATE = 0.10
SEED = 42

TRAIN_CONFIG = TrainConfig(
    hidden_channels=128,
    num_layers=2,
    lr=5e-3,
    weight_decay=1e-5,
    epochs=200,
    patience=30,
    hybrid_alpha=0.8,
    hybrid_beta=0.15,
    hybrid_gamma=0.05,
)


# ─── Synthetic CV generator based on real skill distribution ─────────────────


def generate_cvs_from_real_jobs(
    jobs: list[JobData],
    n: int,
    seed: int = 42,
) -> list[CVData]:
    """Generate synthetic CVs whose skill pool is drawn from real JD skill frequencies.

    This creates more realistic CVs than purely random generation because
    the skill distribution matches what the market actually demands.
    """
    rng = random.Random(seed)

    # Build skill frequency from real jobs
    skill_counter: Counter[str] = Counter()
    for job in jobs:
        skill_counter.update(job.skills)

    # Weighted skill pool
    skill_pool = list(skill_counter.keys())
    skill_weights = [skill_counter[s] for s in skill_pool]

    # Seniority distribution (match real jobs roughly)
    seniority_options = [
        SeniorityLevel.INTERN,
        SeniorityLevel.JUNIOR,
        SeniorityLevel.MID,
        SeniorityLevel.SENIOR,
        SeniorityLevel.LEAD,
        SeniorityLevel.MANAGER,
    ]
    seniority_weights = [0.05, 0.20, 0.35, 0.25, 0.10, 0.05]

    edu_options = [
        EducationLevel.COLLEGE,
        EducationLevel.BACHELOR,
        EducationLevel.MASTER,
        EducationLevel.PHD,
    ]
    edu_weights = [0.10, 0.50, 0.30, 0.10]

    skill_count_range = {
        SeniorityLevel.INTERN: (3, 5),
        SeniorityLevel.JUNIOR: (4, 7),
        SeniorityLevel.MID: (5, 9),
        SeniorityLevel.SENIOR: (7, 12),
        SeniorityLevel.LEAD: (7, 12),
        SeniorityLevel.MANAGER: (6, 10),
    }

    prof_range = {
        SeniorityLevel.INTERN: (1, 2),
        SeniorityLevel.JUNIOR: (1, 3),
        SeniorityLevel.MID: (2, 4),
        SeniorityLevel.SENIOR: (3, 5),
        SeniorityLevel.LEAD: (3, 5),
        SeniorityLevel.MANAGER: (2, 4),
    }

    cvs: list[CVData] = []
    for i in range(n):
        seniority = rng.choices(seniority_options, weights=seniority_weights, k=1)[0]
        yr_lo, yr_hi = SENIORITY_TO_YEARS[seniority]
        experience = round(rng.uniform(yr_lo, yr_hi), 1)
        edu = rng.choices(edu_options, weights=edu_weights, k=1)[0]

        lo, hi = skill_count_range[seniority]
        num_skills = rng.randint(lo, hi)
        # Sample skills weighted by real job frequency
        sampled: list[str] = []
        attempts = 0
        while len(sampled) < num_skills and attempts < num_skills * 10:
            pick = rng.choices(skill_pool, weights=skill_weights, k=1)[0]
            if pick not in sampled:
                sampled.append(pick)
            attempts += 1

        plo, phi = prof_range[seniority]
        proficiencies = [rng.randint(plo, phi) for _ in sampled]

        title = seniority.name.capitalize()
        skill_str = ", ".join(sampled)
        text = (
            f"{title} software engineer with {experience} years of experience. "
            f"Education: {edu.name.lower()}. Skills: {skill_str}."
        )

        cvs.append(
            CVData(
                cv_id=i,
                seniority=seniority,
                experience_years=experience,
                education=edu,
                skills=tuple(sampled),
                skill_proficiencies=tuple(proficiencies),
                text=text,
            )
        )

    return cvs


# ─── Helpers (reused from run_experiment.py) ──────────────────────────────────


def _print_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def _evaluate_baseline(name, scorer, test_pairs, cvs, jobs, cv_map, job_map):
    scores, labels = [], []
    for p in test_pairs:
        ci, ji = cv_map.get(p.cv_id), job_map.get(p.job_id)
        if ci is None or ji is None:
            continue
        scores.append(scorer.score(cvs[ci], jobs[ji]))
        labels.append(p.label)
    if not scores:
        return {}
    return compute_all_metrics(np.array(labels), np.array(scores))


def _print_results_table(results: dict[str, dict[str, float]]) -> None:
    metrics_order = ["recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10", "auc_roc"]
    col_w, name_w = 12, 28
    header = f"{'Method':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metrics_order)
    print("\n" + "=" * (name_w + col_w * len(metrics_order)))
    print(header)
    print("-" * (name_w + col_w * len(metrics_order)))
    best = {}
    for m in metrics_order:
        vals = [(n, r.get(m, 0.0)) for n, r in results.items()]
        if vals:
            best[m] = max(vals, key=lambda x: x[1])[0]
    for name, metrics in results.items():
        row = f"{name:<{name_w}}"
        for m in metrics_order:
            val = metrics.get(m, 0.0)
            marker = " *" if best.get(m) == name else "  "
            row += f"{val:>{col_w - 2}.4f}{marker}"
        print(row)
    print("=" * (name_w + col_w * len(metrics_order)))
    print("  * = best in column")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    t_start = time.time()

    # ─── Step 1: Load real JDs ──────────────────────────────────────────────
    _print_header("Step 1: Load real job postings")
    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    raw_jobs = load_raw_jobs(RAW_JOBS_PATH)
    raw_jobs = deduplicate(raw_jobs)
    logger.info("Loaded %d raw jobs from %s", len(raw_jobs), RAW_JOBS_PATH)

    extractor = SkillExtractor(normalizer)
    all_jobs = extractor.extract_batch(raw_jobs)
    jobs = [j for j in all_jobs if len(j.skills) >= 2]
    logger.info("Extracted %d jobs with >= 2 skills (dropped %d)", len(jobs), len(all_jobs) - len(jobs))

    # Reassign sequential IDs
    jobs = [
        JobData(
            job_id=i, seniority=j.seniority, skills=j.skills,
            skill_importances=j.skill_importances,
            salary_min=j.salary_min, salary_max=j.salary_max, text=j.text,
        )
        for i, j in enumerate(jobs)
    ]

    # ─── Step 2: Generate synthetic CVs from real skill distribution ────────
    _print_header("Step 2: Generate synthetic CVs (skill dist from real JDs)")
    cvs = generate_cvs_from_real_jobs(jobs, n=NUM_CVS, seed=SEED)
    logger.info("Generated %d synthetic CVs", len(cvs))

    # Stats
    all_job_skills = set()
    for j in jobs:
        all_job_skills.update(j.skills)
    all_cv_skills = set()
    for c in cvs:
        all_cv_skills.update(c.skills)
    logger.info("  Real JD unique skills: %d", len(all_job_skills))
    logger.info("  Synthetic CV unique skills: %d", len(all_cv_skills))
    logger.info("  Overlap: %d skills in common", len(all_job_skills & all_cv_skills))

    # ─── Step 3: Label pairs ────────────────────────────────────────────────
    _print_header("Step 3: Label pairs + split")
    labeler = PairLabeler(seed=SEED)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=NUM_POSITIVE_PAIRS, noise_rate=NOISE_RATE)
    dataset = labeler.split(pairs)
    n_pos = sum(1 for p in pairs if p.label == 1)
    n_neg = sum(1 for p in pairs if p.label == 0)
    logger.info("Pairs: %d (pos=%d, neg=%d), noise=%.0f%%", len(pairs), n_pos, n_neg, NOISE_RATE * 100)
    logger.info("Split: train=%d, val=%d, test=%d", len(dataset.train), len(dataset.val), len(dataset.test))

    # ─── Step 4: Build graph ────────────────────────────────────────────────
    _print_header("Step 4: Build embeddings + graph")
    provider = get_provider()
    builder = GraphBuilder(provider)
    t_graph = time.time()
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
    logger.info("Graph built in %.1fs", time.time() - t_graph)
    logger.info("  CV nodes:   %s", data["cv"].x.shape)
    logger.info("  Job nodes:  %s", data["job"].x.shape)
    logger.info("  Skill nodes: %s", data["skill"].x.shape)
    for et in data.edge_types:
        logger.info("  %-45s %d edges", str(et), data[et].edge_index.shape[1])

    # ─── Step 5: Train GNN ──────────────────────────────────────────────────
    _print_header("Step 5: Train GNN (HeteroGraphSAGE + BPR)")
    trainer = Trainer(TRAIN_CONFIG)
    t_train = time.time()
    result = trainer.train(data, dataset, cvs, jobs)
    train_time = time.time() - t_train
    logger.info("Training: %.1fs, best epoch %d/%d, final loss %.4f",
                train_time, result.best_epoch, len(result.train_losses),
                result.train_losses[-1] if result.train_losses else 0)

    # ─── Step 6: Baselines ──────────────────────────────────────────────────
    _print_header("Step 6: Evaluate baselines")
    cv_map = {c.cv_id: i for i, c in enumerate(cvs)}
    job_map = {j.job_id: i for i, j in enumerate(jobs)}

    cosine_m = _evaluate_baseline("Cosine", CosineSimilarityScorer(provider),
                                  dataset.test, cvs, jobs, cv_map, job_map)
    skill_m = _evaluate_baseline("Skill Overlap", SkillOverlapScorer(),
                                 dataset.test, cvs, jobs, cv_map, job_map)
    bm25 = BM25Scorer()
    bm25.fit(cvs)
    bm25_m = _evaluate_baseline("BM25", bm25, dataset.test, cvs, jobs, cv_map, job_map)

    # ─── Step 7: Results ────────────────────────────────────────────────────
    _print_header("Step 7: Results (Real JDs + Synthetic CVs)")
    all_results = {
        "Cosine Similarity": cosine_m,
        "Skill Overlap (Jaccard)": skill_m,
        "BM25": bm25_m,
        "GNN (Hybrid)": result.test_metrics,
    }
    _print_results_table(all_results)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s (training: {train_time:.1f}s)")
    print(f"Data: {len(jobs)} real JDs + {len(cvs)} synthetic CVs")

    gnn_auc = result.test_metrics.get("auc_roc", 0.0)
    best_baseline_auc = max(cosine_m.get("auc_roc", 0), skill_m.get("auc_roc", 0), bm25_m.get("auc_roc", 0))
    diff = gnn_auc - best_baseline_auc
    if diff > 0:
        print(f"\nGNN outperforms best baseline by {diff:+.4f} AUC-ROC")
    else:
        print(f"\nGNN vs best baseline: {diff:+.4f} AUC-ROC")


if __name__ == "__main__":
    main()
