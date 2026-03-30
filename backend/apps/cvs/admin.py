from django.contrib import admin
from apps.cvs.models import CV, CVSkill


class CVSkillInline(admin.TabularInline):
    model = CVSkill
    extra = 0


@admin.register(CV)
class CVAdmin(admin.ModelAdmin):
    list_display = ("__str__", "seniority", "experience_years", "education", "created_at")
    list_filter = ("seniority", "education")
    search_fields = ("parsed_text", "raw_text")
    inlines = [CVSkillInline]
