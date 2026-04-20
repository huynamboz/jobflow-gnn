"""
JobFlow-GNN — Experiment with REAL JDs + REAL CVs (Kaggle)

Real JDs from Indeed (315) + Real CVs from HuggingFace datasetmaster/resumes (4817 IT resumes).
No synthetic data — fully real pipeline.

Usage:
    cd backend
    python run_experiment_real_cv.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ml_service.baselines.bm25 import BM25Scorer
from ml_service.baselines.cosine import CosineSimilarityScorer
from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.data.resume_loader import load_resumes
from ml_service.data.skill_extractor import SkillExtractor
from ml_service.crawler.storage import deduplicate, load_raw_jobs
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.evaluation.metrics import compute_all_metrics
from ml_service.evaluation.per_cv_evaluator import PerCVEvaluator, print_per_cv_results
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import JobData
from ml_service.training.trainer import Trainer, TrainConfig, make_gnn_hybrid_scorer

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"real_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_log_fh = open(_LOG_FILE, "w", encoding="utf-8", buffering=1)  # line-buffered


class _Tee:
    """Write to both stdout and log file so print() is captured."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()
    def isatty(self):
        return False


sys.stdout = _Tee(sys.__stdout__, _log_fh)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.__stdout__),
        logging.StreamHandler(_log_fh),  # reuse same handle, no race condition
    ],
)
logger = logging.getLogger("experiment_real_cv")
logger.info("Log file: %s", _LOG_FILE.resolve())

# ─── Config ───────────────────────────────────────────────────────────────────

RAW_JOBS_PATH = Path("data/raw_jobs.jsonl")
SKILL_ALIAS_PATH = "ml_service/data/skill-alias.json"

CV_SOURCE = "datasetmaster"  # "datasetmaster" (4.8K) or "54k" (54K, IT-filtered)
MAX_CVS = 1000      # cap CVs (full dataset = 4817)
MAX_JOBS = None      # use all crawled JDs
NUM_POSITIVE_PAIRS = 3500
NOISE_RATE = 0.05
SEED = 42

TRAIN_CONFIG = TrainConfig(
    hidden_channels=128,
    num_layers=2,
    lr=5e-3,
    weight_decay=1e-5,
    epochs=200,
    patience=30,
    drop_edge_rate=0.2,
    hybrid_alpha=0.8,
    hybrid_beta=0.15,
    hybrid_gamma=0.05,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


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
    metrics_order = ["recall@5", "recall@10", "precision@5", "precision@10", "hit_rate@5", "hit_rate@10", "mrr", "ndcg@10", "auc_roc"]
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
    _print_header("Step 1: Load real JDs (Indeed)")
    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    raw_jobs = deduplicate(load_raw_jobs(RAW_JOBS_PATH))
    extractor = SkillExtractor(normalizer)
    extractor.fit(raw_jobs)  # compute TF-IDF skill importance
    all_jobs = extractor.extract_batch(raw_jobs)
    jobs = [j for j in all_jobs if len(j.skills) >= 2]
    jobs = [
        JobData(
            job_id=i, seniority=j.seniority, skills=j.skills,
            skill_importances=j.skill_importances,
            salary_min=j.salary_min, salary_max=j.salary_max, text=j.text,
        )
        for i, j in enumerate(jobs)
    ]
    logger.info("Loaded %d real JDs (filtered >= 2 skills)", len(jobs))

    # ─── Step 2: Load real CVs (Kaggle) ─────────────────────────────────────
    _print_header(f"Step 2: Load real CVs (source={CV_SOURCE})")
    cvs = load_resumes(normalizer, source=CV_SOURCE, max_resumes=MAX_CVS)
    logger.info("Loaded %d real CVs", len(cvs))

    # Stats
    all_job_skills = set()
    for j in jobs:
        all_job_skills.update(j.skills)
    all_cv_skills = set()
    for c in cvs:
        all_cv_skills.update(c.skills)
    overlap = all_job_skills & all_cv_skills
    logger.info("  JD unique skills: %d", len(all_job_skills))
    logger.info("  CV unique skills: %d", len(all_cv_skills))
    logger.info("  Overlap: %d skills in common", len(overlap))

    from collections import Counter
    cv_sen = Counter(c.seniority.name for c in cvs)
    job_sen = Counter(j.seniority.name for j in jobs)
    logger.info("  CV seniority: %s", dict(cv_sen))
    logger.info("  JD seniority: %s", dict(job_sen))

    # ─── Step 3: Label pairs ────────────────────────────────────────────────
    _print_header("Step 3: Label pairs + split")
    labeler = PairLabeler(seed=SEED)
    pairs = labeler.create_pairs(
        cvs, jobs, num_positive=NUM_POSITIVE_PAIRS,
        noise_rate=NOISE_RATE, use_skill_relations=True,
    )
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

    # ─── Step 7: Results (Global) ────────────────────────────────────────────
    _print_header("Step 7: Results — Global Evaluation (Real JDs + Real CVs)")
    all_results = {
        "Cosine Similarity": cosine_m,
        "Skill Overlap (Jaccard)": skill_m,
        "BM25": bm25_m,
        "GNN (Hybrid)": result.test_metrics,
    }
    _print_results_table(all_results)

    gnn_auc = result.test_metrics.get("auc_roc", 0.0)
    best_bl_auc = max(cosine_m.get("auc_roc", 0), skill_m.get("auc_roc", 0), bm25_m.get("auc_roc", 0))
    print(f"\nGNN vs best baseline AUC-ROC: {gnn_auc - best_bl_auc:+.4f}")

    gnn_r10 = result.test_metrics.get("recall@10", 0.0)
    best_bl_r10 = max(cosine_m.get("recall@10", 0), skill_m.get("recall@10", 0), bm25_m.get("recall@10", 0))
    print(f"GNN vs best baseline Recall@10: {gnn_r10 - best_bl_r10:+.4f}")

    # ─── Step 8: Per-CV Full-Ranking Evaluation ────────────────────────────
    _print_header("Step 8: Per-CV Full-Ranking Evaluation")
    logger.info("Running per-CV evaluation (each CV ranks ALL %d jobs)...", len(jobs))
    per_cv_eval = PerCVEvaluator(cvs, jobs, dataset, min_test_positives=1)
    per_cv_ks = (10, 20, 50, 100, 200)

    gnn_batch_fn, _ = make_gnn_hybrid_scorer(result.model, result.data_clean, cvs, jobs, TRAIN_CONFIG)
    gnn_per_cv = per_cv_eval.evaluate_batch(gnn_batch_fn, ks=per_cv_ks)

    cosine_per_cv = per_cv_eval.evaluate(CosineSimilarityScorer(provider), ks=per_cv_ks)
    skill_per_cv = per_cv_eval.evaluate(SkillOverlapScorer(), ks=per_cv_ks)
    bm25_per_cv = per_cv_eval.evaluate(bm25, ks=per_cv_ks)

    per_cv_results = {
        "Cosine Similarity": cosine_per_cv,
        "Skill Overlap (Jaccard)": skill_per_cv,
        "BM25": bm25_per_cv,
        "GNN (Hybrid)": gnn_per_cv,
    }
    print_per_cv_results(per_cv_results, ks=per_cv_ks)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s (training: {train_time:.1f}s)")
    print(f"Data: {len(jobs)} real JDs (Indeed) + {len(cvs)} real CVs (Kaggle)")


if __name__ == "__main__":
    main()
