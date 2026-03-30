from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ml_service.baselines.skill_overlap import SkillOverlapScorer
from ml_service.evaluation.metrics import compute_all_metrics
from ml_service.graph.schema import (
    EDGE_TRIPLETS,
    CVData,
    DatasetSplit,
    EdgeType,
    JobData,
    LabeledPair,
)
from ml_service.models.gnn import HeteroGraphSAGE, HeteroRGCN, prepare_data_for_gnn
from ml_service.models.losses import bpr_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_type: str = "graphsage"  # "graphsage" or "rgcn"
    hidden_channels: int = 128
    num_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    hybrid_alpha: float = 0.6
    hybrid_beta: float = 0.3
    hybrid_gamma: float = 0.1


@dataclass
class TrainResult:
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_metrics_history: list[dict[str, float]] = field(default_factory=list)
    test_metrics: dict[str, float] = field(default_factory=dict)


def _strip_label_edges(data: HeteroData) -> HeteroData:
    """Remove match/no_match edges to prevent label leakage during GNN message passing."""
    data = copy.copy(data)
    for etype in (EdgeType.MATCH, EdgeType.NO_MATCH):
        triplet = EDGE_TRIPLETS[etype]
        if triplet in data.edge_types:
            del data[triplet]
    return data


def _sample_bpr_pairs(
    pairs: list[LabeledPair],
    rng: np.random.RandomState,
    cv_id_to_idx: dict[int, int],
    job_id_to_idx: dict[int, int],
    num_jobs: int,
    *,
    hard_neg_ratio: float = 0.7,
    cvs: list[CVData] | None = None,
    jobs: list[JobData] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (cv, pos_job, neg_job) BPR triplets with hard negative priority.

    Hard negatives (same role, similar skills, wrong match) are sampled
    with probability `hard_neg_ratio`. This forces GNN to learn subtle
    differences like Flask vs React, not just Frontend vs DevOps.
    """
    pos_by_cv: dict[int, list[int]] = {}
    neg_by_cv: dict[int, list[int]] = {}
    for p in pairs:
        if p.cv_id not in cv_id_to_idx or p.job_id not in job_id_to_idx:
            continue
        if p.label == 1:
            pos_by_cv.setdefault(p.cv_id, []).append(job_id_to_idx[p.job_id])
        else:
            neg_by_cv.setdefault(p.cv_id, []).append(job_id_to_idx[p.job_id])

    # Split negatives into hard vs easy (if CV/job data available)
    hard_neg_by_cv: dict[int, list[int]] = {}
    easy_neg_by_cv: dict[int, list[int]] = {}

    if cvs and jobs:
        cv_idx_to_data = {cv_id_to_idx.get(cv.cv_id): cv for cv in cvs if cv.cv_id in cv_id_to_idx}
        job_idx_to_data = {job_id_to_idx.get(j.job_id): j for j in jobs if j.job_id in job_id_to_idx}

        for cv_id, neg_jobs in neg_by_cv.items():
            cv_idx = cv_id_to_idx[cv_id]
            cv_data = cv_idx_to_data.get(cv_idx)
            if not cv_data:
                easy_neg_by_cv[cv_id] = neg_jobs
                continue

            cv_skills = set(cv_data.skills)
            hard, easy = [], []
            for job_idx in neg_jobs:
                job_data = job_idx_to_data.get(job_idx)
                if job_data:
                    job_skills = set(job_data.skills)
                    union = cv_skills | job_skills
                    overlap = len(cv_skills & job_skills) / len(union) if union else 0
                    sen_dist = abs(int(cv_data.seniority) - int(job_data.seniority))
                    # Hard: some overlap + similar seniority (subtle mismatch)
                    if overlap >= 0.15 and sen_dist <= 1:
                        hard.append(job_idx)
                    else:
                        easy.append(job_idx)
                else:
                    easy.append(job_idx)

            hard_neg_by_cv[cv_id] = hard
            easy_neg_by_cv[cv_id] = easy
    else:
        for cv_id, neg_jobs in neg_by_cv.items():
            easy_neg_by_cv[cv_id] = neg_jobs

    cv_indices, pos_indices, neg_indices = [], [], []
    for cv_id, pos_jobs in pos_by_cv.items():
        cv_idx = cv_id_to_idx[cv_id]
        hard_negs = hard_neg_by_cv.get(cv_id, [])
        easy_negs = easy_neg_by_cv.get(cv_id, [])
        all_negs = hard_negs + easy_negs

        for pos_job in pos_jobs:
            # Prefer hard negatives with probability hard_neg_ratio
            if hard_negs and rng.random() < hard_neg_ratio:
                neg_job = hard_negs[rng.randint(0, len(hard_negs))]
            elif all_negs:
                neg_job = all_negs[rng.randint(0, len(all_negs))]
            else:
                neg_job = rng.randint(0, num_jobs)

            cv_indices.append(cv_idx)
            pos_indices.append(pos_job)
            neg_indices.append(neg_job)

    return (
        torch.tensor(cv_indices, dtype=torch.long),
        torch.tensor(pos_indices, dtype=torch.long),
        torch.tensor(neg_indices, dtype=torch.long),
    )


def _seniority_match_score(cv: CVData, job: JobData) -> float:
    """1.0 if seniority matches, decaying by distance."""
    dist = abs(int(cv.seniority) - int(job.seniority))
    return max(0.0, 1.0 - dist * 0.25)


class Trainer:
    """BPR training loop with early stopping and hybrid scoring evaluation."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()

    def train(
        self,
        data: HeteroData,
        dataset: DatasetSplit,
        cvs: list[CVData],
        jobs: list[JobData],
    ) -> TrainResult:
        """Train the GNN model and return results."""
        cfg = self.config
        rng = np.random.RandomState(42)

        cv_id_to_idx = {cv.cv_id: i for i, cv in enumerate(cvs)}
        job_id_to_idx = {job.job_id: i for i, job in enumerate(jobs)}

        # Strip label edges and prepare data for GNN
        data_clean = _strip_label_edges(data)
        data_clean = prepare_data_for_gnn(data_clean)

        # Build model
        metadata = data_clean.metadata()
        if cfg.model_type == "rgcn":
            model = HeteroRGCN(
                metadata=metadata,
                hidden_channels=cfg.hidden_channels,
                num_layers=cfg.num_layers,
            )
        else:
            model = HeteroGraphSAGE(
                metadata=metadata,
                hidden_channels=cfg.hidden_channels,
                num_layers=cfg.num_layers,
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        result = TrainResult()
        best_val_mrr = -1.0
        patience_counter = 0
        best_state = None

        for epoch in range(cfg.epochs):
            # --- Train ---
            model.train()
            cv_idx, pos_idx, neg_idx = _sample_bpr_pairs(
                dataset.train, rng, cv_id_to_idx, job_id_to_idx, len(jobs),
                cvs=cvs, jobs=jobs,
            )
            if len(cv_idx) == 0:
                logger.warning("No BPR pairs sampled for epoch %d", epoch)
                result.train_losses.append(0.0)
                result.val_metrics_history.append({})
                continue

            z_dict = model.encode(data_clean)
            pos_scores = model.decode(z_dict, cv_idx, pos_idx)
            neg_scores = model.decode(z_dict, cv_idx, neg_idx)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            result.train_losses.append(loss_val)

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                val_metrics = self._evaluate_split(
                    model, data_clean, dataset.val, cvs, jobs, cv_id_to_idx, job_id_to_idx
                )
            result.val_metrics_history.append(val_metrics)

            val_mrr = val_metrics.get("mrr", 0.0)
            logger.info("Epoch %d — loss=%.4f, val_mrr=%.4f", epoch, loss_val, val_mrr)

            # Early stopping
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_state = copy.deepcopy(model.state_dict())
                result.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best model and evaluate on test
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            result.test_metrics = self._evaluate_split(
                model, data_clean, dataset.test, cvs, jobs, cv_id_to_idx, job_id_to_idx
            )

        return result

    def _evaluate_split(
        self,
        model: HeteroGraphSAGE,
        data: HeteroData,
        pairs: list[LabeledPair],
        cvs: list[CVData],
        jobs: list[JobData],
        cv_id_to_idx: dict[int, int],
        job_id_to_idx: dict[int, int],
    ) -> dict[str, float]:
        """Evaluate using hybrid scoring: alpha*GNN + beta*skill_overlap + gamma*seniority."""
        if not pairs:
            return {}

        cfg = self.config
        skill_scorer = SkillOverlapScorer()

        cv_indices = []
        job_indices = []
        valid_pairs = []
        for p in pairs:
            ci = cv_id_to_idx.get(p.cv_id)
            ji = job_id_to_idx.get(p.job_id)
            if ci is not None and ji is not None:
                cv_indices.append(ci)
                job_indices.append(ji)
                valid_pairs.append(p)

        if not valid_pairs:
            return {}

        cv_idx_t = torch.tensor(cv_indices, dtype=torch.long)
        job_idx_t = torch.tensor(job_indices, dtype=torch.long)

        # GNN scores (min-max normalized to [0, 1])
        z_dict = model.encode(data)
        gnn_scores = model.decode(z_dict, cv_idx_t, job_idx_t).cpu().numpy()
        gnn_min, gnn_max = gnn_scores.min(), gnn_scores.max()
        if gnn_max - gnn_min > 1e-8:
            gnn_norm = (gnn_scores - gnn_min) / (gnn_max - gnn_min)
        else:
            gnn_norm = np.full_like(gnn_scores, 0.5)

        # Feature-based scores
        skill_scores = np.array(
            [skill_scorer.score(cvs[ci], jobs[ji]) for ci, ji in zip(cv_indices, job_indices)]
        )
        seniority_scores = np.array(
            [_seniority_match_score(cvs[ci], jobs[ji]) for ci, ji in zip(cv_indices, job_indices)]
        )

        # Hybrid combination
        hybrid = (
            cfg.hybrid_alpha * gnn_norm
            + cfg.hybrid_beta * skill_scores
            + cfg.hybrid_gamma * seniority_scores
        )

        y_true = np.array([p.label for p in valid_pairs])
        return compute_all_metrics(y_true, hybrid)
