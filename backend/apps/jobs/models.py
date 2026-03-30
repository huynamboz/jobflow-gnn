from django.db import models


class Job(models.Model):
    class Seniority(models.IntegerChoices):
        INTERN = 0
        JUNIOR = 1
        MID = 2
        SENIOR = 3
        LEAD = 4
        MANAGER = 5

    # Core fields
    title = models.CharField(max_length=500)
    company = models.CharField(max_length=300, blank=True, default="")
    location = models.CharField(max_length=300, blank=True, default="")
    description = models.TextField()
    seniority = models.IntegerField(choices=Seniority.choices, default=Seniority.MID)

    # Salary
    salary_min = models.IntegerField(default=0)
    salary_max = models.IntegerField(default=0)
    salary_currency = models.CharField(max_length=10, default="USD")

    # Source
    source = models.CharField(max_length=50, blank=True, default="")  # indeed, adzuna, etc.
    source_url = models.URLField(max_length=1000, blank=True, default="")

    # Skills (M2M through JobSkill)
    skills = models.ManyToManyField("skills.Skill", through="JobSkill", blank=True)

    # Timestamps
    date_posted = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "jobs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} @ {self.company}"


class JobSkill(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="job_skills")
    skill = models.ForeignKey("skills.Skill", on_delete=models.CASCADE)
    importance = models.IntegerField(default=3)  # 1-5

    class Meta:
        db_table = "job_skills"
        unique_together = ("job", "skill")
