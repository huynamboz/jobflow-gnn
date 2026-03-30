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

    # Parsed fields
    seniority = models.IntegerField(choices=Seniority.choices, default=Seniority.MID)
    experience_years = models.FloatField(default=0.0)
    education = models.IntegerField(choices=Education.choices, default=Education.BACHELOR)
    parsed_text = models.TextField(blank=True, default="")  # text for embedding

    # File
    file = models.FileField(upload_to="cvs/", null=True, blank=True)
    raw_text = models.TextField(blank=True, default="")  # extracted text from PDF/DOCX

    # Skills (M2M through CVSkill)
    skills = models.ManyToManyField("skills.Skill", through="CVSkill", blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "cvs"
        ordering = ["-created_at"]

    def __str__(self):
        name = self.user.get_full_name() if self.user else f"CV #{self.pk}"
        return f"{name} ({self.get_seniority_display()})"


class CVSkill(models.Model):
    cv = models.ForeignKey(CV, on_delete=models.CASCADE, related_name="cv_skills")
    skill = models.ForeignKey("skills.Skill", on_delete=models.CASCADE)
    proficiency = models.IntegerField(default=3)  # 1-5

    class Meta:
        db_table = "cv_skills"
        unique_together = ("cv", "skill")
