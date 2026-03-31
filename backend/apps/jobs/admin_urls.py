"""Admin-only job management URLs (/api/admin/...)."""

from django.urls import path

from apps.jobs.admin_views import (
    AdminCompanyDetailView,
    AdminCompanyListView,
    AdminJobDetailView,
    AdminJobListView,
    AdminPlatformDetailView,
    AdminPlatformListView,
)

urlpatterns = [
    path("platforms/", AdminPlatformListView.as_view(), name="admin-platform-list"),
    path("platforms/<int:pk>/", AdminPlatformDetailView.as_view(), name="admin-platform-detail"),
    path("companies/", AdminCompanyListView.as_view(), name="admin-company-list"),
    path("companies/<int:pk>/", AdminCompanyDetailView.as_view(), name="admin-company-detail"),
    path("jobs/", AdminJobListView.as_view(), name="admin-job-list"),
    path("jobs/<int:pk>/", AdminJobDetailView.as_view(), name="admin-job-detail"),
]
