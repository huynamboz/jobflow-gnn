from django.contrib import admin

from apps.jobs.models import Company, CompanyPlatform, Job, JobSkill, Platform


@admin.register(Platform)
class PlatformAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "is_active", "job_count", "created_at")
    list_filter = ("is_active",)
    search_fields = ("name",)
    prepopulated_fields = {"slug": ("name",)}

    def job_count(self, obj):
        return obj.jobs.count()
    job_count.short_description = "Jobs"


class CompanyPlatformInline(admin.TabularInline):
    model = CompanyPlatform
    extra = 0


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ("name", "industry", "size", "location", "job_count")
    list_filter = ("industry",)
    search_fields = ("name",)
    inlines = [CompanyPlatformInline]

    def job_count(self, obj):
        return obj.jobs.count()
    job_count.short_description = "Jobs"


class JobSkillInline(admin.TabularInline):
    model = JobSkill
    extra = 0


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("title", "company", "platform", "seniority", "job_type", "is_active", "created_at")
    list_filter = ("platform", "seniority", "job_type", "is_active")
    search_fields = ("title", "description", "company__name")
    raw_id_fields = ("company", "platform")
    inlines = [JobSkillInline]
