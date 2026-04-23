from django.conf import settings
from django.db import models


class CV(models.Model):
    class Seniority(models.IntegerChoices):
        INTERN = 0
        JUNIOR = 1
        MID = 2
        SENIOR = 3
        LEAD = 4
        MANAGER = 5

    class Education(models.IntegerChoices):
        NONE = 0
        COLLEGE = 1
        BACHELOR = 2
        MASTER = 3
        PHD = 4

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="cvs",
        null=True, blank=True,
    )

    # File
    file = models.FileField(upload_to="cvs/", null=True, blank=True)
    file_name = models.CharField(max_length=300, blank=True, default="")

    # Raw + parsed text
    raw_text = models.TextField(blank=True, default="")  # extracted from PDF/DOCX
    parsed_text = models.TextField(blank=True, default="")  # cleaned text for embedding

    # Parsed fields
    candidate_name = models.CharField(max_length=200, blank=True, default="")
    seniority = models.IntegerField(choices=Seniority.choices, default=Seniority.MID)
    experience_years = models.FloatField(default=0.0)
    education = models.IntegerField(choices=Education.choices, default=Education.BACHELOR)
    role_category = models.CharField(max_length=20, blank=True, default="other")
    work_experience = models.JSONField(default=list, blank=True)

    # Skills (M2M through CVSkill)
    skills = models.ManyToManyField("skills.Skill", through="CVSkill", blank=True)

    # Source (for dataset CVs)
    source = models.CharField(max_length=50, blank=True, default="")  # "upload", "linkedin_dataset", "kaggle"
    source_category = models.CharField(max_length=100, blank=True, default="")  # "Software Engineer", "AI", etc.

    # Status
    is_active = models.BooleanField(default=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "cvs"
        ordering = ["-created_at"]

    def __str__(self):
        if self.user:
            return f"{self.user.get_full_name() or self.user.username} ({self.get_seniority_display()})"
        return f"CV #{self.pk} ({self.get_seniority_display()})"


class CVSkill(models.Model):
    cv = models.ForeignKey(CV, on_delete=models.CASCADE, related_name="cv_skills")
    skill = models.ForeignKey("skills.Skill", on_delete=models.CASCADE, related_name="cv_skills")
    proficiency = models.IntegerField(default=3)  # 1-5

    class Meta:
        db_table = "cv_skills"
        unique_together = ("cv", "skill")


class CVExtractionBatch(models.Model):
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_ERROR = "error"
    STATUS_CANCELLED = "cancelled"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_DONE, "Done"),
        (STATUS_ERROR, "Error"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    # Filter used to select CVs for this batch
    filter_source = models.CharField(max_length=50, blank=True, default="")
    filter_source_categories = models.JSONField(default=list, blank=True)  # e.g. ["AI", "Devops"]

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    total = models.IntegerField(default=0)
    done_count = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "cv_extraction_batches"
        ordering = ["-created_at"]

    def __str__(self):
        return f"CVBatch #{self.id} ({self.status})"


class CVExtractionRecord(models.Model):
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_DONE = "done"
    STATUS_ERROR = "error"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_DONE, "Done"),
        (STATUS_ERROR, "Error"),
    ]

    batch = models.ForeignKey(CVExtractionBatch, on_delete=models.CASCADE, related_name="records")
    cv = models.ForeignKey(CV, on_delete=models.CASCADE, related_name="extraction_records")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    result = models.JSONField(null=True, blank=True)
    error_msg = models.CharField(max_length=500, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "cv_extraction_records"
        ordering = ["cv_id"]
