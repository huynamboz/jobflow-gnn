"""Retrain GNN model from DB data.

Usage: python manage.py retrain_model
"""

from django.core.management.base import BaseCommand

from apps.matching.services import TrainService


class Command(BaseCommand):
    help = "Retrain GNN model from database data"

    def handle(self, *args, **options):
        self.stdout.write("Starting model training...")

        try:
            run = TrainService.run_training()
            self.stdout.write(self.style.SUCCESS(
                f"Training completed: AUC={run.auc_roc:.4f}, "
                f"epoch={run.best_epoch}, "
                f"{run.num_jobs} jobs, {run.num_cvs} CVs"
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Training failed: {e}"))
