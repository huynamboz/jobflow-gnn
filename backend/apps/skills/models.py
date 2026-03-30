from django.db import models


class Skill(models.Model):
    class Category(models.IntegerChoices):
        TECHNICAL = 0
        SOFT = 1
        TOOL = 2
        DOMAIN = 3

    canonical_name = models.CharField(max_length=100, unique=True, db_index=True)
    category = models.IntegerField(choices=Category.choices, default=Category.TECHNICAL)
    aliases = models.JSONField(default=list, blank=True)

    class Meta:
        db_table = "skills"
        ordering = ["canonical_name"]

    def __str__(self):
        return self.canonical_name
