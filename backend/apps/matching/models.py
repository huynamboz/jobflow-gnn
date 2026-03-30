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
