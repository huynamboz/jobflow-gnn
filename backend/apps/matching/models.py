from django.db import models


class MatchResult(models.Model):
    cv = models.ForeignKey("cvs.CV", on_delete=models.CASCADE, related_name="matches")
    job = models.ForeignKey("jobs.Job", on_delete=models.CASCADE, related_name="matches")
    score = models.FloatField()
    eligible = models.BooleanField(default=False)
    matched_skills = models.JSONField(default=list)
    missing_skills = models.JSONField(default=list)
    seniority_match = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "match_results"
        ordering = ["-score"]
        unique_together = ("cv", "job")

    def __str__(self):
        return f"CV {self.cv_id} × Job {self.job_id} = {self.score:.2f}"


class Feedback(models.Model):
    class Rating(models.TextChoices):
        RELEVANT = "relevant"
        NOT_RELEVANT = "not_relevant"

    match = models.ForeignKey(MatchResult, on_delete=models.CASCADE, related_name="feedbacks")
    user = models.ForeignKey("users.User", on_delete=models.CASCADE)
    rating = models.CharField(max_length=20, choices=Rating.choices)
    comment = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "feedback"
        ordering = ["-created_at"]


class TrainRun(models.Model):
    """Track model training history with versioning."""

    class Status(models.TextChoices):
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    # Version
    version = models.CharField(max_length=50, unique=True)  # "v1", "v2", auto-generated
    is_active = models.BooleanField(default=False)  # admin chooses which model is live
    description = models.TextField(blank=True, default="")  # admin notes

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.RUNNING)

    # Data stats
    num_jobs = models.IntegerField(default=0)
    num_cvs = models.IntegerField(default=0)
    num_pairs = models.IntegerField(default=0)
    num_skills = models.IntegerField(default=0)

    # Benchmark metrics
    auc_roc = models.FloatField(null=True, blank=True)
    recall_at_5 = models.FloatField(null=True, blank=True)
    recall_at_10 = models.FloatField(null=True, blank=True)
    precision_at_5 = models.FloatField(null=True, blank=True)
    precision_at_10 = models.FloatField(null=True, blank=True)
    ndcg_at_10 = models.FloatField(null=True, blank=True)
    mrr = models.FloatField(null=True, blank=True)
    best_epoch = models.IntegerField(null=True, blank=True)
    final_loss = models.FloatField(null=True, blank=True)
    reranker_accuracy = models.FloatField(null=True, blank=True)
    metrics_json = models.JSONField(default=dict, blank=True)  # full metrics

    # Training config
    model_type = models.CharField(max_length=50, default="graphsage")  # graphsage, rgcn
    hidden_channels = models.IntegerField(default=256)
    num_layers = models.IntegerField(default=3)
    learning_rate = models.FloatField(default=1e-3)
    config_json = models.JSONField(default=dict, blank=True)  # full config

    # Checkpoint
    checkpoint_path = models.CharField(max_length=500, blank=True, default="")

    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    training_duration_seconds = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "train_runs"
        ordering = ["-started_at"]

    def __str__(self):
        active = " [ACTIVE]" if self.is_active else ""
        return f"{self.version} ({self.status}) AUC={self.auc_roc or 0:.3f}{active}"

    def save(self, *args, **kwargs):
        if not self.version:
            last = TrainRun.objects.order_by("-id").first()
            num = (last.id + 1) if last else 1
            self.version = f"v{num}"
        super().save(*args, **kwargs)

    def activate(self):
        """Set this run as active, deactivate others."""
        TrainRun.objects.filter(is_active=True).update(is_active=False)
        self.is_active = True
        self.save(update_fields=["is_active"])
