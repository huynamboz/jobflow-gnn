from django.contrib import admin

from .models import HumanLabel, LabelingCV, LabelingJob, PairQueue


@admin.register(LabelingCV)
class LabelingCVAdmin(admin.ModelAdmin):
    list_display = ["cv_id", "seniority", "experience_years", "education", "source", "created_at"]
    search_fields = ["cv_id"]
    list_filter = ["seniority", "education", "source"]


@admin.register(LabelingJob)
class LabelingJobAdmin(admin.ModelAdmin):
    list_display = ["job_id", "title", "seniority", "salary_min", "salary_max", "created_at"]
    search_fields = ["job_id", "title"]
    list_filter = ["seniority"]


@admin.register(PairQueue)
class PairQueueAdmin(admin.ModelAdmin):
    list_display = ["id", "cv", "job", "selection_reason", "status", "split", "skill_overlap_score", "priority"]
    list_filter = ["status", "selection_reason", "split"]
    search_fields = ["cv__cv_id", "job__job_id"]


@admin.register(HumanLabel)
class HumanLabelAdmin(admin.ModelAdmin):
    list_display = ["id", "pair", "overall", "skill_fit", "seniority_fit", "experience_fit", "domain_fit", "labeled_by", "created_at"]
    list_filter = ["overall"]
    search_fields = ["pair__cv__cv_id", "pair__job__job_id"]
