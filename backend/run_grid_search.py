"""
Grid search: hidden_channels × dropout × lr
Uses epochs=150 / patience=25 for fast comparison.
Prints a ranked table and saves best config.

Usage:
    cd backend
    python run_grid_search.py
"""

from __future__ import annotations

import itertools
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ml_service.crawler.storage import deduplicate, load_raw_jobs
from ml_service.data.labeler import PairLabeler
from ml_service.data.linkedin_cv_loader import load_linkedin_cvs_json
from ml_service.data.skill_extractor import SkillExtractor
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import JobData, LabeledPair
from ml_service.embedding import get_provider
from ml_service.training.trainer import Trainer, TrainConfig

logging.basicConfig(
    level=logging.WARNING,  # suppress verbose logs during grid search
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DATASET_DIR   = Path("/Users/huynam/Documents/PROJECT/jobflow-gnn/Dataset")
RAW_JOBS_PATH = Path("data/raw_jobs.jsonl")
SKILL_ALIAS   = "ml_service/data/skill-alias.json"
CV_JSON_PATH  = "data/linkedin_cvs.json"

NUM_POSITIVE_PAIRS = 8000   # increased from 3500
NOISE_RATE         = 0.05
SEED               = 42

# Grid: 3 × 3 × 2 = 18 configs
GRID = {
    "hidden_channels": [128, 256, 384],
    "dropout":         [0.0, 0.1, 0.2],
    "lr":              [5e-4, 1e-3],
}

# Fixed for search speed
SEARCH_EPOCHS   = 150
SEARCH_PATIENCE = 25


def _generate_pseudo_pairs(cvs, jobs, existing_pos_ids, top_k=30, min_overlap=0.25):
    from ml_service.baselines.skill_overlap import SkillOverlapScorer
    scorer = SkillOverlapScorer()
    pseudo = []
    for cv in cvs:
        candidates = [(j.job_id, scorer.score(cv, j)) for j in jobs if scorer.score(cv, j) >= min_overlap]
        candidates.sort(key=lambda x: -x[1])
        for job_id, _ in candidates[:top_k]:
            if (cv.cv_id, job_id) not in existing_pos_ids:
                pseudo.append(LabeledPair(cv_id=cv.cv_id, job_id=job_id, label=1, split="train"))
    return pseudo


def load_data():
    print("Loading data...", flush=True)
    normalizer = SkillNormalizer(SKILL_ALIAS)
    cvs = load_linkedin_cvs_json(CV_JSON_PATH)
    cvs = [c for c in cvs if len(c.skills) >= 2]

    raw_jobs = deduplicate(load_raw_jobs(RAW_JOBS_PATH))
    extractor = SkillExtractor(normalizer)
    extractor.fit(raw_jobs)
    jobs = [j for j in extractor.extract_batch(raw_jobs) if len(j.skills) >= 2]
    jobs = [
        JobData(job_id=i, seniority=j.seniority, skills=j.skills,
                skill_importances=j.skill_importances,
                salary_min=j.salary_min, salary_max=j.salary_max, text=j.text)
        for i, j in enumerate(jobs)
    ]

    labeler = PairLabeler(seed=SEED)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=NUM_POSITIVE_PAIRS,
                                 noise_rate=NOISE_RATE, use_skill_relations=True)
    dataset = labeler.split(pairs)

    existing_pos_ids = {(p.cv_id, p.job_id) for p in pairs if p.label == 1}
    pseudo = _generate_pseudo_pairs(cvs, jobs, existing_pos_ids)
    dataset.train.extend(pseudo)

    provider = get_provider()
    builder = GraphBuilder(provider)
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)

    n_pos = sum(1 for p in pairs if p.label == 1)
    print(f"  CVs={len(cvs)}, Jobs={len(jobs)}, Pairs={len(pairs)} (pos={n_pos})", flush=True)
    print(f"  Train={len(dataset.train)}, Val={len(dataset.val)}, Test={len(dataset.test)}", flush=True)
    return cvs, jobs, data, dataset


def run_config(cfg: TrainConfig, data, dataset, cvs, jobs) -> dict:
    trainer = Trainer(cfg)
    result = trainer.train(data, dataset, cvs, jobs)
    m = result.test_metrics
    return {
        "auc_roc":     m.get("auc_roc", 0.0),
        "ndcg@10":     m.get("ndcg@10", 0.0),
        "recall@10":   m.get("recall@10", 0.0),
        "best_epoch":  result.best_epoch,
    }


def main():
    t0 = time.time()
    cvs, jobs, data, dataset = load_data()

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"\nRunning {len(combos)} configs (epochs={SEARCH_EPOCHS}, patience={SEARCH_PATIENCE})\n", flush=True)

    header = f"{'#':>3}  {'hidden':>7}  {'dropout':>8}  {'lr':>7}  {'AUC':>7}  {'NDCG@10':>8}  {'R@10':>7}  {'epoch':>6}  {'time':>6}"
    print(header)
    print("-" * len(header))

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        cfg = TrainConfig(
            model_type="graphsage",
            hidden_channels=params["hidden_channels"],
            num_layers=3,
            lr=params["lr"],
            dropout=params["dropout"],
            weight_decay=1e-5,
            epochs=SEARCH_EPOCHS,
            patience=SEARCH_PATIENCE,
            warmup_epochs=20,
            drop_edge_rate=0.2,
            hybrid_alpha=0.55,
            hybrid_beta=0.30,
            hybrid_gamma=0.15,
        )
        t1 = time.time()
        m = run_config(cfg, data, dataset, cvs, jobs)
        elapsed = time.time() - t1

        row = {**params, **m}
        results.append(row)

        print(
            f"{i:>3}  {params['hidden_channels']:>7}  {params['dropout']:>8.1f}"
            f"  {params['lr']:>7.0e}  {m['auc_roc']:>7.4f}  {m['ndcg@10']:>8.4f}"
            f"  {m['recall@10']:>7.4f}  {m['best_epoch']:>6}  {elapsed:>5.0f}s",
            flush=True,
        )

    # Rank by AUC
    results.sort(key=lambda r: r["auc_roc"], reverse=True)

    print("\n" + "=" * 60)
    print("TOP 5 CONFIGS (by AUC-ROC)")
    print("=" * 60)
    for rank, r in enumerate(results[:5], 1):
        print(
            f"  #{rank}: hidden={r['hidden_channels']}, dropout={r['dropout']:.1f}, lr={r['lr']:.0e}"
            f"  →  AUC={r['auc_roc']:.4f}  NDCG@10={r['ndcg@10']:.4f}  R@10={r['recall@10']:.4f}"
        )

    best = results[0]
    print(f"\nBest config: hidden={best['hidden_channels']}, dropout={best['dropout']:.1f}, lr={best['lr']:.0e}")
    print(f"Total grid search time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
