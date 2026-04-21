"""
Experiment: Real JDs (Indeed) + Real LinkedIn CVs (870 Vietnamese IT professionals)

Usage:
    cd backend
    python run_experiment_linkedin_cv.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ml_service.baselines.bm25 import BM25Scorer
from ml_service.baselines.cosine import CosineSimilarityScorer
from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.crawler.storage import deduplicate, load_raw_jobs
from ml_service.data.labeler import PairLabeler
from ml_service.data.linkedin_cv_loader import load_linkedin_cvs_json
from ml_service.data.skill_extractor import SkillExtractor
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.evaluation.metrics import compute_all_metrics
from ml_service.evaluation.per_cv_evaluator import PerCVEvaluator, print_per_cv_results
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import JobData
from ml_service.graph.schema import LabeledPair
from ml_service.training.trainer import Trainer, TrainConfig, make_gnn_hybrid_scorer

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"linkedin_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
logger = logging.getLogger("experiment_linkedin")
logger.info("Log file: %s", _LOG_FILE.resolve())

CV_JSON_PATH  = Path("data/linkedin_cvs.json")   # JSON cache — fast, no PDF parsing needed
RAW_JOBS_PATH = Path("data/raw_jobs.jsonl")
SKILL_ALIAS_PATH = "ml_service/data/skill-alias.json"

NUM_POSITIVE_PAIRS = 8000   # increased from 3500 — uses 80% of available positives
NOISE_RATE = 0.05
SEED = 42

# Best from grid search: hidden=384, dropout=0.1, lr=1e-3 (NDCG=1.0, R@10 best)
# Runner-up by AUC: hidden=256, dropout=0.1, lr=1e-3 (AUC=0.7369)
TRAIN_CONFIG = TrainConfig(
    model_type="graphsage",
    hidden_channels=384,
    num_layers=3,
    lr=1e-3,
    dropout=0.1,
    weight_decay=1e-5,
    epochs=300,
    patience=50,
    warmup_epochs=30,
    drop_edge_rate=0.2,
    hybrid_alpha=0.55,
    hybrid_beta=0.30,
    hybrid_gamma=0.15,
)


def _generate_pseudo_pairs(
    cvs, jobs,
    existing_pos_ids: set[tuple[int, int]],
    top_k: int = 30,
    min_overlap: float = 0.25,
) -> list[LabeledPair]:
    """Pseudo-positive pairs using Skill Overlap as teacher signal (Fix 2).

    For each CV, rank all jobs by Skill Overlap and take top-K as pseudo-positives.
    This expands training coverage from 0.44% → ~30x more, teaching the GNN to
    rank skill-matched jobs above unrelated ones across the full job space.
    Only added to training set — test/val stay unchanged.
    """
    from ml_service.baselines.skill_overlap import SkillOverlapScorer
    scorer = SkillOverlapScorer()
    pseudo: list[LabeledPair] = []
    for cv in cvs:
        candidates = []
        for job in jobs:
            s = scorer.score(cv, job)
            if s >= min_overlap:
                candidates.append((job.job_id, s))
        candidates.sort(key=lambda x: -x[1])
        for job_id, _ in candidates[:top_k]:
            if (cv.cv_id, job_id) not in existing_pos_ids:
                pseudo.append(LabeledPair(cv_id=cv.cv_id, job_id=job_id, label=1, split="train"))
    return pseudo


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
    cvs = load_linkedin_cvs_json(CV_JSON_PATH)
    cvs = [c for c in cvs if len(c.skills) >= 2]
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

    # Fix 2: Pseudo-positive expansion — add Skill Overlap top-K to train only
    # Test/val remain clean (proxy-labeled ground truth only)
    existing_pos_ids = {(p.cv_id, p.job_id) for p in pairs if p.label == 1}
    pseudo_pairs = _generate_pseudo_pairs(cvs, jobs, existing_pos_ids, top_k=30, min_overlap=0.25)
    dataset.train.extend(pseudo_pairs)
    logger.info("Pseudo-positive pairs added to train: %d (total train: %d)", len(pseudo_pairs), len(dataset.train))

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

    # --- Step 7: Results (Global) ---
    _print_header("Step 7: Results — Global Evaluation (Real JDs + LinkedIn CVs)")
    all_results = {
        "Cosine Similarity": cosine_m,
        "Skill Overlap (Jaccard)": skill_m,
        "BM25": bm25_m,
        "GNN (Hybrid)": result.test_metrics,
    }
    _print_results_table(all_results)

    # --- Step 8: Per-CV Evaluation (Full-ranking + 2-stage) ---
    _print_header("Step 8: Per-CV Evaluation")
    per_cv_eval = PerCVEvaluator(cvs, jobs, dataset, min_test_positives=1)
    per_cv_ks = (10, 20, 50, 100)
    gnn_batch_fn, _ = make_gnn_hybrid_scorer(result.model, result.data_clean, cvs, jobs, TRAIN_CONFIG)
    skill_scorer = SkillOverlapScorer()

    # 8a: Full-ranking baseline (each CV ranks ALL jobs) — honest upper-bound check
    _print_header("Step 8a: Full-Ranking (all %d jobs)" % len(jobs))
    gnn_full    = per_cv_eval.evaluate_batch(gnn_batch_fn, ks=per_cv_ks)
    skill_full  = per_cv_eval.evaluate(skill_scorer, ks=per_cv_ks)
    bm25_full   = per_cv_eval.evaluate(bm25, ks=per_cv_ks)
    cosine_full = per_cv_eval.evaluate(CosineSimilarityScorer(provider), ks=per_cv_ks)

    print_per_cv_results({
        "Cosine (full)":         cosine_full,
        "Skill Overlap (full)":  skill_full,
        "BM25 (full)":           bm25_full,
        "GNN (full)":            gnn_full,
    }, ks=per_cv_ks)

    # 8b: 2-stage pipeline — phản ánh đúng cách system hoạt động trong thực tế
    # Stage 1: Skill Overlap retrieve top-100 → Stage 2: GNN re-rank
    _print_header("Step 8b: 2-Stage Pipeline (Stage1=SkillOverlap top-100, Stage2=GNN)")
    gnn_2stage = per_cv_eval.evaluate_twostage(
        stage1_scorer=skill_scorer,
        stage2_batch_fn=gnn_batch_fn,
        retrieve_n=100,
        ks=per_cv_ks,
    )
    skill_2stage = per_cv_eval.evaluate_twostage(
        stage1_scorer=skill_scorer,
        stage2_batch_fn=lambda cv, jobs: np.array([skill_scorer.score(cv, j) for j in jobs]),
        retrieve_n=100,
        ks=per_cv_ks,
    )
    bm25_2stage = per_cv_eval.evaluate_twostage(
        stage1_scorer=skill_scorer,
        stage2_batch_fn=lambda cv, jobs: np.array([bm25.score(cv, j) for j in jobs]),
        retrieve_n=100,
        ks=per_cv_ks,
    )

    print_per_cv_results({
        "Skill Overlap (2-stage)": skill_2stage,
        "BM25 (2-stage)":          bm25_2stage,
        "GNN (2-stage)":           gnn_2stage,
    }, ks=per_cv_ks)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Data: {len(jobs)} real JDs + {len(cvs)} LinkedIn CVs")


if __name__ == "__main__":
    main()
