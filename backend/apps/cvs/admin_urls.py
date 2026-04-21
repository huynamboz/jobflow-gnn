"""Admin-only CV management URLs (/api/admin/cvs/)."""

from django.urls import path

from apps.cvs.admin_views import AdminCVDetailView, AdminCVListView

urlpatterns = [
    path("cvs/", AdminCVListView.as_view(), name="admin-cv-list"),
    path("cvs/<int:pk>/", AdminCVDetailView.as_view(), name="admin-cv-detail"),
]
