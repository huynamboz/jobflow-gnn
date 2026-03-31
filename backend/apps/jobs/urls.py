from django.urls import path

from apps.jobs.views import JobDetailView, JobListView

urlpatterns = [
    path("", JobListView.as_view(), name="job-list"),
    path("<int:pk>/", JobDetailView.as_view(), name="job-detail"),
]
