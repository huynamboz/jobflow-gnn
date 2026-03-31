"""Training service: DB → Graph → Train GNN → Save checkpoint."""

from __future__ import annotations

import logging
import time

from django.conf import settings
from django.utils import timezone

from apps.cvs.services import CVService
from apps.jobs.services import JobService
from apps.matching.models import TrainRun

logger = logging.getLogger(__name__)


class TrainService:
    """Orchestrate model training from DB data."""

    @staticmethod
    def run_training() -> TrainRun:
        """Full training pipeline: DB → graph → GNN → reranker → checkpoint."""
        from ml_service.data.labeler import PairLabeler
        from ml_service.data.skill_normalization import SkillNormalizer
        from ml_service.embedding import get_provider
        from ml_service.graph.builder import GraphBuilder
        from ml_service.training.trainer import Trainer, TrainConfig

        run = TrainRun.objects.create(status=TrainRun.Status.RUNNING)
        t_start = time.time()

        try:
            # Load data from DB
            logger.info("Loading data from DB...")
            jobs = JobService.get_all_job_data()
            cvs = CVService.get_all_cv_data()

            if not jobs or not cvs:
                raise ValueError(f"Not enough data: {len(jobs)} jobs, {len(cvs)} CVs")

            run.num_jobs = len(jobs)
            run.num_cvs = len(cvs)

            # Label pairs
            labeler = PairLabeler(seed=42)
            pairs = labeler.create_pairs(cvs, jobs, num_positive=min(2000, len(cvs) * 5), noise_rate=0.10, use_skill_relations=True)
            dataset = labeler.split(pairs)
            run.num_pairs = len(pairs)

            # Build graph
            normalizer = SkillNormalizer()
            provider = get_provider()
            builder = GraphBuilder(provider)
            data = builder.build(cvs, jobs, normalizer.skill_catalog, pairs)
            run.num_skills = data["skill"].x.shape[0]

            # Train GNN
            config = TrainConfig(
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
            trainer = Trainer(config)
            result = trainer.train(data, dataset, cvs, jobs)

            # Save checkpoint
            from ml_service.inference.checkpoint import save_checkpoint
            from ml_service.models.gnn import HeteroGraphSAGE, prepare_data_for_gnn
            from ml_service.training.trainer import _strip_label_edges, _sample_bpr_pairs
            from ml_service.models.losses import bpr_loss
            import copy, numpy as np, torch

            # Rebuild model for checkpoint
            data_clean = _strip_label_edges(data)
            data_clean = prepare_data_for_gnn(data_clean)
            model = HeteroGraphSAGE(
                metadata=data_clean.metadata(),
                hidden_channels=config.hidden_channels,
                num_layers=config.num_layers,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            rng = np.random.RandomState(42)
            cv_map = {cv.cv_id: i for i, cv in enumerate(cvs)}
            job_map = {j.job_id: i for i, j in enumerate(jobs)}

            best_state = None
            for epoch in range(result.best_epoch + 1):
                model.train()
                ci, pi, ni = _sample_bpr_pairs(dataset.train, rng, cv_map, job_map, len(jobs), cvs=cvs, jobs=jobs)
                if len(ci) == 0:
                    continue
                z = model.encode(data_clean)
                loss = bpr_loss(model.decode(z, ci, pi), model.decode(z, ci, ni))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                best_state = copy.deepcopy(model.state_dict())

            if best_state:
                model.load_state_dict(best_state)
            model.eval()

            checkpoint_path = settings.ML_CHECKPOINT_DIR
            save_checkpoint(checkpoint_path, model, data, cvs, jobs, metadata={
                "best_epoch": result.best_epoch,
                "test_metrics": result.test_metrics,
                "train_config": {"hidden_channels": config.hidden_channels, "num_layers": config.num_layers},
            })

            # Update run
            run.status = TrainRun.Status.COMPLETED
            run.auc_roc = result.test_metrics.get("auc_roc")
            run.best_epoch = result.best_epoch
            run.final_loss = result.train_losses[-1] if result.train_losses else None
            run.metrics_json = result.test_metrics
            run.config_json = {"hidden_channels": config.hidden_channels, "num_layers": config.num_layers, "lr": config.lr}
            run.checkpoint_path = checkpoint_path
            run.completed_at = timezone.now()
            run.save()

            elapsed = time.time() - t_start
            logger.info("Training completed in %.1fs: AUC=%.4f", elapsed, run.auc_roc or 0)

        except Exception as e:
            run.status = TrainRun.Status.FAILED
            run.metrics_json = {"error": str(e)}
            run.completed_at = timezone.now()
            run.save()
            logger.error("Training failed: %s", e)
            raise

        return run
