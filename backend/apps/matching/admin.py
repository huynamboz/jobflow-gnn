from django.contrib import admin
from apps.matching.models import Feedback, MatchResult


@admin.register(MatchResult)
class MatchResultAdmin(admin.ModelAdmin):
    list_display = ("cv", "job", "score", "eligible", "seniority_match", "created_at")
    list_filter = ("eligible", "seniority_match")
    search_fields = ("cv__parsed_text", "job__title")


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("match", "user", "rating", "created_at")
    list_filter = ("rating",)
