"""
JobFlow-GNN — Full Experiment Pipeline

Generates synthetic data, builds graph, trains GNN, runs baselines,
and prints a comparison table.

Usage:
    cd ml-service
    python run_experiment.py
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ml_service.baselines.bm25 import BM25Scorer
from ml_service.baselines.cosine import CosineSimilarityScorer
from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.data.generator import SyntheticDataGenerator
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.evaluation.metrics import compute_all_metrics
from ml_service.graph.builder import GraphBuilder
from ml_service.training.trainer import Trainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("experiment")


# ─── Config ───────────────────────────────────────────────────────────────────

SKILL_ALIAS_PATH = "../roadmap/week1/skill-alias.json"
NUM_CVS = 300
NUM_JOBS = 600
NUM_POSITIVE_PAIRS = 800
SEED = 42

# Noise settings — create asymmetry between labels and baselines
SYNONYM_RATE = 0.3          # 30% of skills in text use synonyms
IMPLICIT_SKILL_RATE = 0.2   # 20% of CV skills only appear in text (not in skills tuple)
CLUSTER_RATE = 0.3          # 30% of jobs have cluster requirements
NOISE_RATE = 0.12           # 12% label noise

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


def _print_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def _evaluate_baseline(
    name: str,
    scorer,
    test_pairs,
    cvs,
    jobs,
    cv_id_to_idx,
    job_id_to_idx,
) -> dict[str, float]:
    """Score all test pairs with a baseline scorer and compute metrics."""
    scores = []
    labels = []
    for p in test_pairs:
        ci = cv_id_to_idx.get(p.cv_id)
        ji = job_id_to_idx.get(p.job_id)
        if ci is None or ji is None:
            continue
        scores.append(scorer.score(cvs[ci], jobs[ji]))
        labels.append(p.label)

    if not scores:
        return {}

    y_true = np.array(labels)
    y_scores = np.array(scores)
    metrics = compute_all_metrics(y_true, y_scores)
    return metrics


def _print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table."""
    metrics_order = ["recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10", "auc_roc"]

    # Header
    col_w = 12
    name_w = 28
    header = f"{'Method':<{name_w}}"
    for m in metrics_order:
        header += f"{m:>{col_w}}"
    print("\n" + "=" * (name_w + col_w * len(metrics_order)))
    print(header)
    print("-" * (name_w + col_w * len(metrics_order)))

    # Find best value per metric
    best = {}
    for m in metrics_order:
        vals = [(name, r.get(m, 0.0)) for name, r in results.items()]
        if vals:
            best[m] = max(vals, key=lambda x: x[1])[0]

    # Rows
    for name, metrics in results.items():
        row = f"{name:<{name_w}}"
        for m in metrics_order:
            val = metrics.get(m, 0.0)
            marker = " *" if best.get(m) == name else "  "
            row += f"{val:>{col_w - 2}.4f}{marker}"
        print(row)

    print("=" * (name_w + col_w * len(metrics_order)))
    print("  * = best in column")


def main() -> None:
    t_start = time.time()

    # ─── Step 1: Load skill normalizer ──────────────────────────────────────
    _print_header("Step 1: Load skill normalizer")
    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    logger.info("Loaded %d canonical skills", len(normalizer.canonical_skills))

    # ─── Step 2: Generate synthetic data ────────────────────────────────────
    _print_header("Step 2: Generate synthetic data (with noise)")
    gen = SyntheticDataGenerator(
        normalizer,
        seed=SEED,
        synonym_rate=SYNONYM_RATE,
        implicit_skill_rate=IMPLICIT_SKILL_RATE,
        cluster_rate=CLUSTER_RATE,
    )
    cvs = gen.generate_cvs(NUM_CVS)
    jobs = gen.generate_jobs(NUM_JOBS)
    n_text_skills = sum(1 for v in gen.cv_text_skills.values() if v)
    n_clustered_jobs = len(gen.job_clusters)
    logger.info("Generated %d CVs, %d JDs", len(cvs), len(jobs))
    logger.info("  CVs with text-only skills: %d (%.0f%%)", n_text_skills, 100 * n_text_skills / len(cvs))
    logger.info("  Jobs with cluster requirements: %d (%.0f%%)", n_clustered_jobs, 100 * n_clustered_jobs / len(jobs))

    # ─── Step 3: Create labeled pairs + split ───────────────────────────────
    _print_header("Step 3: Label pairs + train/val/test split")
    labeler = PairLabeler(seed=SEED)
    pairs = labeler.create_pairs(
        cvs, jobs, num_positive=NUM_POSITIVE_PAIRS,
        cv_text_skills=gen.cv_text_skills,
        job_clusters=gen.job_clusters,
        noise_rate=NOISE_RATE,
    )
    dataset = labeler.split(pairs)
    n_pos = sum(1 for p in pairs if p.label == 1)
    n_neg = sum(1 for p in pairs if p.label == 0)
    logger.info("Total pairs: %d (positive=%d, negative=%d)", len(pairs), n_pos, n_neg)
    logger.info("  Label noise rate: %.0f%%", NOISE_RATE * 100)
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(dataset.train), len(dataset.val), len(dataset.test),
    )

    # ─── Step 4: Build embedding + graph ────────────────────────────────────
    _print_header("Step 4: Build embeddings + heterogeneous graph")
    provider = get_provider()
    logger.info("Embedding provider: %s (dim=%d)", type(provider).__name__, provider.dim)

    builder = GraphBuilder(provider)
    t_graph = time.time()
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
    logger.info("Graph built in %.1fs", time.time() - t_graph)
    logger.info("  CV nodes:        %s", data["cv"].x.shape)
    logger.info("  Job nodes:       %s", data["job"].x.shape)
    logger.info("  Skill nodes:     %s", data["skill"].x.shape)
    logger.info("  Seniority nodes: %s", data["seniority"].x.shape)
    for et in data.edge_types:
        logger.info("  %-45s %d edges", str(et), data[et].edge_index.shape[1])

    # ─── Step 5: Train GNN ──────────────────────────────────────────────────
    _print_header("Step 5: Train GNN (HeteroGraphSAGE + BPR loss)")
    trainer = Trainer(TRAIN_CONFIG)
    t_train = time.time()
    result = trainer.train(data, dataset, cvs, jobs)
    train_time = time.time() - t_train
    logger.info("Training completed in %.1fs", train_time)
    logger.info("Best epoch: %d / %d", result.best_epoch, len(result.train_losses))
    logger.info("Final train loss: %.4f", result.train_losses[-1] if result.train_losses else 0)

    # ─── Step 6: Evaluate baselines on test set ─────────────────────────────
    _print_header("Step 6: Evaluate baselines on test set")
    cv_id_to_idx = {cv.cv_id: i for i, cv in enumerate(cvs)}
    job_id_to_idx = {job.job_id: i for i, job in enumerate(jobs)}

    # Baseline 1: Cosine Similarity
    logger.info("Running Cosine Similarity baseline...")
    cosine_scorer = CosineSimilarityScorer(provider)
    cosine_metrics = _evaluate_baseline(
        "Cosine Similarity", cosine_scorer,
        dataset.test, cvs, jobs, cv_id_to_idx, job_id_to_idx,
    )

    # Baseline 2: Skill Overlap (Jaccard)
    logger.info("Running Skill Overlap baseline...")
    skill_scorer = SkillOverlapScorer()
    skill_metrics = _evaluate_baseline(
        "Skill Overlap", skill_scorer,
        dataset.test, cvs, jobs, cv_id_to_idx, job_id_to_idx,
    )

    # Baseline 3: BM25
    logger.info("Running BM25 baseline...")
    bm25_scorer = BM25Scorer()
    bm25_scorer.fit(cvs)
    bm25_metrics = _evaluate_baseline(
        "BM25", bm25_scorer,
        dataset.test, cvs, jobs, cv_id_to_idx, job_id_to_idx,
    )

    # ─── Step 7: Print comparison table ─────────────────────────────────────
    _print_header("Step 7: Results comparison")

    all_results = {
        "Cosine Similarity": cosine_metrics,
        "Skill Overlap (Jaccard)": skill_metrics,
        "BM25": bm25_metrics,
        "GNN (Hybrid)": result.test_metrics,
    }

    _print_results_table(all_results)

    # ─── Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"Training time: {train_time:.1f}s")

    # Check if GNN beats baselines
    gnn_mrr = result.test_metrics.get("mrr", 0.0)
    best_baseline_mrr = max(
        cosine_metrics.get("mrr", 0.0),
        skill_metrics.get("mrr", 0.0),
        bm25_metrics.get("mrr", 0.0),
    )
    diff = gnn_mrr - best_baseline_mrr
    if diff > 0:
        print(f"\nGNN outperforms best baseline by {diff:+.4f} MRR")
    else:
        print(f"\nGNN underperforms best baseline by {diff:+.4f} MRR — needs tuning")


if __name__ == "__main__":
    main()
