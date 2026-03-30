from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class Role(models.TextChoices):
        CANDIDATE = "candidate"
        RECRUITER = "recruiter"
        ADMIN = "admin"

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.CANDIDATE)

    class Meta:
        db_table = "users"
