"""
Experiment: Real JDs (Indeed) + Real LinkedIn CVs (870 Vietnamese IT professionals)

Usage:
    cd backend
    python run_experiment_linkedin_cv.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ml_service.baselines.bm25 import BM25Scorer
from ml_service.baselines.cosine import CosineSimilarityScorer
from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.crawler.storage import deduplicate, load_raw_jobs
from ml_service.data.labeler import PairLabeler
from ml_service.data.linkedin_cv_loader import load_linkedin_cvs
from ml_service.data.skill_extractor import SkillExtractor
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.evaluation.metrics import compute_all_metrics
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import JobData
from ml_service.training.trainer import Trainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("experiment_linkedin")

DATASET_DIR = Path("/Users/huynam/Documents/PROJECT/jobflow-gnn/Dataset")
RAW_JOBS_PATH = Path("data/raw_jobs.jsonl")
SKILL_ALIAS_PATH = "ml_service/data/skill-alias.json"

NUM_POSITIVE_PAIRS = 2000
NOISE_RATE = 0.10
SEED = 42

TRAIN_CONFIG = TrainConfig(
    model_type="graphsage",
    hidden_channels=256,
    num_layers=3,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=300,
    patience=50,
    hybrid_alpha=0.55,
    hybrid_beta=0.30,
    hybrid_gamma=0.15,
)


def _print_header(title):
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


def _print_results_table(results):
    metrics_order = ["recall@5", "recall@10", "precision@5", "precision@10", "hit_rate@10", "mrr", "ndcg@10", "auc_roc"]
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


def main():
    t_start = time.time()

    # --- Step 1: Load real JDs ---
    _print_header("Step 1: Load real JDs")
    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
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
    logger.info("Loaded %d real JDs", len(jobs))

    # --- Step 2: Load LinkedIn CVs ---
    _print_header("Step 2: Load LinkedIn CVs")
    cvs = load_linkedin_cvs(
        DATASET_DIR, normalizer,
        min_skills=2,
        categories=["AI", "Devops", "Software Engineer", "Tester", "Business Analyst", "UX_UI"],
    )
    logger.info("Loaded %d LinkedIn CVs", len(cvs))

    from collections import Counter
    cv_skills = Counter()
    for c in cvs:
        cv_skills.update(c.skills)
    jd_skills = set()
    for j in jobs:
        jd_skills.update(j.skills)
    overlap = set(cv_skills.keys()) & jd_skills
    logger.info("  CV unique skills: %d", len(cv_skills))
    logger.info("  JD unique skills: %d", len(jd_skills))
    logger.info("  Overlap: %d", len(overlap))
    logger.info("  Avg skills/CV: %.1f", sum(cv_skills.values()) / max(len(cvs), 1))

    sen_dist = Counter(c.seniority.name for c in cvs)
    logger.info("  Seniority: %s", dict(sen_dist))

    # --- Step 3: Label pairs ---
    _print_header("Step 3: Label pairs + split")
    labeler = PairLabeler(seed=SEED)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=NUM_POSITIVE_PAIRS, noise_rate=NOISE_RATE, use_skill_relations=True)
    dataset = labeler.split(pairs)
    n_pos = sum(1 for p in pairs if p.label == 1)
    n_neg = sum(1 for p in pairs if p.label == 0)
    logger.info("Pairs: %d (pos=%d, neg=%d)", len(pairs), n_pos, n_neg)
    logger.info("Split: train=%d, val=%d, test=%d", len(dataset.train), len(dataset.val), len(dataset.test))

    # --- Step 4: Build graph ---
    _print_header("Step 4: Build graph")
    provider = get_provider()
    builder = GraphBuilder(provider)
    t_graph = time.time()
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
    logger.info("Graph built in %.1fs", time.time() - t_graph)
    for et in data.edge_types:
        logger.info("  %-45s %d edges", str(et), data[et].edge_index.shape[1])

    # --- Step 5: Train GNN ---
    _print_header("Step 5: Train GNN")
    trainer = Trainer(TRAIN_CONFIG)
    t_train = time.time()
    result = trainer.train(data, dataset, cvs, jobs)
    logger.info("Training: %.1fs, best epoch %d, final loss %.4f",
                time.time() - t_train, result.best_epoch,
                result.train_losses[-1] if result.train_losses else 0)

    # --- Step 6: Baselines ---
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

    # --- Step 7: Results ---
    _print_header("Step 7: Results (Real JDs + LinkedIn CVs)")
    all_results = {
        "Cosine Similarity": cosine_m,
        "Skill Overlap (Jaccard)": skill_m,
        "BM25": bm25_m,
        "GNN (Hybrid)": result.test_metrics,
    }
    _print_results_table(all_results)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Data: {len(jobs)} real JDs + {len(cvs)} LinkedIn CVs")


if __name__ == "__main__":
    main()
