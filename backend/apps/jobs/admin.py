from django.contrib import admin
from apps.jobs.models import Job, JobSkill


class JobSkillInline(admin.TabularInline):
    model = JobSkill
    extra = 0


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("title", "company", "seniority", "source", "created_at")
    list_filter = ("seniority", "source")
    search_fields = ("title", "company", "description")
    inlines = [JobSkillInline]
