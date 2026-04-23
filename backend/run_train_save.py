"""
Train GNN from exported DB dataset and save checkpoint for inference.

Usage:
    cd backend
    python export_dataset.py --output data/processed/v2   # first: export from DB
    python run_train_save.py                               # then: train
    python run_train_save.py --data data/processed/v1
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ml_service.embedding import get_provider
from ml_service.graph.builder import GraphBuilder
from ml_service.graph.schema import (
    CVData,
    DatasetSplit,
    EducationLevel,
    JobData,
    LabeledPair,
    SeniorityLevel,
    SkillCategory,
)
from ml_service.inference.checkpoint import save_checkpoint
from ml_service.models.gnn import HeteroGraphSAGE, prepare_data_for_gnn
from ml_service.models.losses import bpr_loss
from ml_service.training.trainer import TrainConfig, Trainer, _sample_bpr_pairs, _strip_label_edges

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_save")

CHECKPOINT_DIR = Path("checkpoints/latest")

TRAIN_CONFIG = TrainConfig(
    model_type="graphsage",
    hidden_channels=256,
    num_layers=3,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=300,
    patience=80,
    hybrid_alpha=0.55,
    hybrid_beta=0.30,
    hybrid_gamma=0.15,
)


def load_dataset(data_dir: Path):
    """Load CVData, JobData, LabeledPair lists and skill_catalog from exported JSON files."""

    def jload(name):
        with open(data_dir / name, encoding="utf-8") as f:
            return json.load(f)

    raw_cvs    = jload("cvs.json")
    raw_jobs   = jload("jobs.json")
    raw_labels = jload("labels.json")
    raw_skills = jload("skills.json")

    # Skill catalog: name → SkillCategory
    skill_catalog: dict[str, SkillCategory] = {
        s["name"]: SkillCategory(s["category"]) for s in raw_skills
    }

    # CVData — cv_id = sequential idx from export (matches label cv_idx)
    cvs: list[CVData] = []
    for c in raw_cvs:
        cvs.append(CVData(
            cv_id=c["idx"],
            seniority=SeniorityLevel(c["seniority"]),
            experience_years=float(c["experience_years"]),
            education=EducationLevel(c["education"]),
            skills=tuple(c["skills"]),
            skill_proficiencies=tuple(c["skill_proficiencies"]),
            text=c["text"] or "",
        ))

    # JobData — job_id = sequential idx from export (matches label job_idx)
    jobs: list[JobData] = []
    for j in raw_jobs:
        jobs.append(JobData(
            job_id=j["idx"],
            seniority=SeniorityLevel(j["seniority"]),
            skills=tuple(j["skills"]),
            skill_importances=tuple(j["skill_importances"]),
            salary_min=int(j["salary_min"] or 0),
            salary_max=int(j["salary_max"] or 0),
            text=j["text"] or "",
        ))

    # LabeledPair — cv_id/job_id match the sequential indices above
    split_map: dict[str, list[LabeledPair]] = {"train": [], "val": [], "test": []}
    for lbl in raw_labels:
        split = lbl.get("split", "train")
        split_map.setdefault(split, []).append(
            LabeledPair(
                cv_id=lbl["cv_idx"],
                job_id=lbl["job_idx"],
                label=lbl["label"],
                split=split,
            )
        )

    all_pairs = split_map["train"] + split_map.get("val", []) + split_map.get("test", [])
    dataset = DatasetSplit(
        train=split_map["train"],
        val=split_map.get("val", []),
        test=split_map.get("test", []),
    )

    pos = sum(1 for p in all_pairs if p.label == 1)
    logger.info(
        "Dataset: %d CVs, %d Jobs, %d pairs (%d pos / %d neg)",
        len(cvs), len(jobs), len(all_pairs), pos, len(all_pairs) - pos,
    )
    logger.info(
        "Split: %d train / %d val / %d test",
        len(dataset.train), len(dataset.val), len(dataset.test),
    )
    return cvs, jobs, all_pairs, dataset, skill_catalog


def main(data_dir: Path) -> None:
    t_start = time.time()

    # --- Load data ---
    logger.info("Loading dataset from %s ...", data_dir)
    cvs, jobs, pairs, dataset, skill_catalog = load_dataset(data_dir)

    # --- Build graph ---
    logger.info("Building graph ...")
    provider = get_provider()
    builder = GraphBuilder(provider)
    data = builder.build(cvs, jobs, skill_catalog, pairs)
    logger.info(
        "Graph: %d CVs, %d Jobs, %d Skills",
        data["cv"].x.shape[0], data["job"].x.shape[0], data["skill"].x.shape[0],
    )

    # --- Train ---
    logger.info("Training GNN ...")
    trainer = Trainer(TRAIN_CONFIG)
    result = trainer.train(data, dataset, cvs, jobs)
    logger.info("Best epoch: %d, final loss: %.4f", result.best_epoch, result.train_losses[-1])
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in result.test_metrics.items()})

    # --- Rebuild model with best weights for checkpoint ---
    logger.info("Rebuilding model for checkpoint ...")
    cfg = TRAIN_CONFIG
    data_clean   = _strip_label_edges(data)
    data_prepared = prepare_data_for_gnn(data_clean)

    model = HeteroGraphSAGE(
        metadata=data_prepared.metadata(),
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.RandomState(42)
    cv_id_to_idx  = {cv.cv_id:   i for i, cv  in enumerate(cvs)}
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

    # --- Save checkpoint ---
    save_checkpoint(
        CHECKPOINT_DIR, model, data, cvs, jobs,
        metadata={
            "best_epoch":   result.best_epoch,
            "test_metrics": result.test_metrics,
            "num_cvs":      len(cvs),
            "num_jobs":     len(jobs),
            "cv_source":    "db_export",
            "data_dir":     str(data_dir),
            "train_config": {
                "hidden_channels": cfg.hidden_channels,
                "num_layers":      cfg.num_layers,
                "lr":              cfg.lr,
                "hybrid_alpha":    cfg.hybrid_alpha,
                "hybrid_beta":     cfg.hybrid_beta,
                "hybrid_gamma":    cfg.hybrid_gamma,
            },
        },
    )

    total = time.time() - t_start
    logger.info("Done in %.1fs. Checkpoint: %s", total, CHECKPOINT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/v2", help="Dataset directory")
    args = parser.parse_args()
    main(Path(args.data))
