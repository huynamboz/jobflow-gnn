"""
Train GNN on real JDs + real CVs and save checkpoint for inference.

Usage:
    cd ml-service
    python run_train_save.py
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ml_service.crawler.resume_loader import load_resumes
from ml_service.crawler.skill_extractor import SkillExtractor
from ml_service.crawler.storage import deduplicate, load_raw_jobs
from ml_service.data.labeler import PairLabeler
from ml_service.data.skill_normalization import SkillNormalizer
from ml_service.embedding import get_provider
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import JobData
from ml_service.inference.checkpoint import save_checkpoint
from ml_service.models.gnn import HeteroGraphSAGE, prepare_data_for_gnn
from ml_service.models.losses import bpr_loss
from ml_service.training.trainer import TrainConfig, Trainer, _strip_label_edges, _sample_bpr_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_save")

RAW_JOBS_PATH = Path("data/raw_jobs.jsonl")
SKILL_ALIAS_PATH = "../roadmap/week1/skill-alias.json"
CHECKPOINT_DIR = Path("checkpoints/latest")

CV_SOURCE = "datasetmaster"
MAX_CVS = 1000
NUM_POSITIVE_PAIRS = 1500
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


def main() -> None:
    t_start = time.time()

    # --- Load data ---
    logger.info("Loading real JDs...")
    normalizer = SkillNormalizer(SKILL_ALIAS_PATH)
    raw_jobs = deduplicate(load_raw_jobs(RAW_JOBS_PATH))
    extractor = SkillExtractor(normalizer)
    all_jobs = extractor.extract_batch(raw_jobs)
    jobs = [j for j in all_jobs if len(j.skills) >= 2]
    jobs = [
        JobData(job_id=i, seniority=j.seniority, skills=j.skills,
                skill_importances=j.skill_importances,
                salary_min=j.salary_min, salary_max=j.salary_max, text=j.text)
        for i, j in enumerate(jobs)
    ]
    logger.info("Loaded %d real JDs", len(jobs))

    logger.info("Loading real CVs (source=%s, max=%d)...", CV_SOURCE, MAX_CVS)
    cvs = load_resumes(normalizer, source=CV_SOURCE, max_resumes=MAX_CVS)
    logger.info("Loaded %d real CVs", len(cvs))

    # --- Label pairs ---
    labeler = PairLabeler(seed=SEED)
    pairs = labeler.create_pairs(cvs, jobs, num_positive=NUM_POSITIVE_PAIRS, noise_rate=NOISE_RATE)
    dataset = labeler.split(pairs)
    logger.info("Pairs: %d, split: %d/%d/%d",
                len(pairs), len(dataset.train), len(dataset.val), len(dataset.test))

    # --- Build graph ---
    logger.info("Building graph...")
    provider = get_provider()
    builder = GraphBuilder(provider)
    data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
    logger.info("Graph: %d CVs, %d Jobs, %d Skills",
                data["cv"].x.shape[0], data["job"].x.shape[0], data["skill"].x.shape[0])

    # --- Train (via Trainer) ---
    logger.info("Training GNN...")
    trainer = Trainer(TRAIN_CONFIG)
    result = trainer.train(data, dataset, cvs, jobs)
    logger.info("Best epoch: %d, final loss: %.4f", result.best_epoch, result.train_losses[-1])
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in result.test_metrics.items()})

    # --- Rebuild model with best weights for checkpoint ---
    logger.info("Rebuilding model for checkpoint...")
    cfg = TRAIN_CONFIG
    data_clean = _strip_label_edges(data)
    data_prepared = prepare_data_for_gnn(data_clean)

    model = HeteroGraphSAGE(
        metadata=data_prepared.metadata(),
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.RandomState(42)
    cv_id_to_idx = {cv.cv_id: i for i, cv in enumerate(cvs)}
    job_id_to_idx = {job.job_id: i for i, job in enumerate(jobs)}

    best_state = None
    for epoch in range(result.best_epoch + 1):
        model.train()
        cv_idx, pos_idx, neg_idx = _sample_bpr_pairs(
            dataset.train, rng, cv_id_to_idx, job_id_to_idx, len(jobs)
        )
        if len(cv_idx) == 0:
            continue
        z = model.encode(data_prepared)
        pos_s = model.decode(z, cv_idx, pos_idx)
        neg_s = model.decode(z, cv_idx, neg_idx)
        loss = bpr_loss(pos_s, neg_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # --- Save ---
    save_checkpoint(
        CHECKPOINT_DIR, model, data, cvs, jobs,
        metadata={
            "best_epoch": result.best_epoch,
            "test_metrics": result.test_metrics,
            "num_jds": len(jobs),
            "num_cvs": len(cvs),
            "cv_source": CV_SOURCE,
            "train_config": {
                "hidden_channels": cfg.hidden_channels,
                "num_layers": cfg.num_layers,
                "lr": cfg.lr,
                "hybrid_alpha": cfg.hybrid_alpha,
                "hybrid_beta": cfg.hybrid_beta,
                "hybrid_gamma": cfg.hybrid_gamma,
            },
        },
    )

    total = time.time() - t_start
    logger.info("Done in %.1fs. Checkpoint: %s", total, CHECKPOINT_DIR)
    logger.info("Data: %d real JDs + %d real CVs (%s)", len(jobs), len(cvs), CV_SOURCE)


if __name__ == "__main__":
    main()
