from django.db import models


class LabelingCV(models.Model):
    cv_id            = models.IntegerField(unique=True)
    source           = models.CharField(max_length=50, default="linkedin")
    skills           = models.JSONField(default=list)
    seniority        = models.CharField(max_length=20, default="MID")
    experience_years = models.FloatField(default=0.0)
    education        = models.CharField(max_length=20, default="BACHELOR")
    text_summary     = models.TextField(default="")
    pdf_path         = models.CharField(max_length=500, blank=True, default="")
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "labeling_cv"

    def __str__(self):
        return f"CV#{self.cv_id} ({self.seniority})"


class LabelingJob(models.Model):
    job_id       = models.IntegerField(unique=True)
    title        = models.CharField(max_length=200, default="")
    skills       = models.JSONField(default=list)
    seniority    = models.CharField(max_length=20, default="MID")
    salary_min   = models.IntegerField(null=True, blank=True)
    salary_max   = models.IntegerField(null=True, blank=True)
    text_summary = models.TextField(default="")
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "labeling_job"

    def __str__(self):
        return f"Job#{self.job_id}: {self.title}"


class SelectionReason(models.TextChoices):
    MEDIUM_OVERLAP = "medium_overlap", "Medium Overlap"
    HIGH_OVERLAP   = "high_overlap",   "High Overlap"
    HARD_NEGATIVE  = "hard_negative",  "Hard Negative"
    RANDOM         = "random",         "Random"


class PairStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    LABELED = "labeled", "Labeled"
    SKIPPED = "skipped", "Skipped"


REASON_PRIORITY = {
    SelectionReason.MEDIUM_OVERLAP: 1,
    SelectionReason.HIGH_OVERLAP:   2,
    SelectionReason.HARD_NEGATIVE:  3,
    SelectionReason.RANDOM:         4,
}


class PairQueue(models.Model):
    cv                  = models.ForeignKey(LabelingCV, on_delete=models.CASCADE, related_name="pairs")
    job                 = models.ForeignKey(LabelingJob, on_delete=models.CASCADE, related_name="pairs")
    skill_overlap_score = models.FloatField(default=0.0)
    bm25_score          = models.FloatField(default=0.0)
    selection_reason    = models.CharField(max_length=30, choices=SelectionReason.choices)
    priority            = models.IntegerField(default=4)
    status              = models.CharField(max_length=20, choices=PairStatus.choices, default=PairStatus.PENDING)
    split               = models.CharField(max_length=10, default="train")
    created_at          = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "labeling_pair_queue"
        unique_together = ("cv", "job")
        ordering = ["priority", "cv_id"]

    def __str__(self):
        return f"Pair CV#{self.cv_id}–Job#{self.job_id} [{self.status}]"


class LabelValue(models.IntegerChoices):
    NOT_SUITABLE = 0, "Không phù hợp"
    SUITABLE     = 1, "Phù hợp"
    STRONG_FIT   = 2, "Rất phù hợp"


class DimScore(models.IntegerChoices):
    BAD  = 0, "Không phù hợp"
    OK   = 1, "Tạm được"
    GOOD = 2, "Tốt"


class HumanLabel(models.Model):
    pair           = models.ForeignKey(PairQueue, on_delete=models.CASCADE, related_name="labels")
    skill_fit      = models.IntegerField(choices=DimScore.choices)
    seniority_fit  = models.IntegerField(choices=DimScore.choices)
    experience_fit = models.IntegerField(choices=DimScore.choices)
    domain_fit     = models.IntegerField(choices=DimScore.choices)
    overall        = models.IntegerField(choices=LabelValue.choices)
    note           = models.TextField(blank=True, default="")
    labeled_by     = models.ForeignKey("users.User", on_delete=models.SET_NULL, null=True, blank=True)
    created_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "labeling_human_label"

    @property
    def binary_label(self) -> int:
        return 1 if self.overall >= 1 else 0

    def __str__(self):
        return f"Label for Pair#{self.pair_id} overall={self.overall}"
