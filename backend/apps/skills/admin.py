from django.contrib import admin
from apps.skills.models import Skill


@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = ("canonical_name", "category")
    list_filter = ("category",)
    search_fields = ("canonical_name",)
