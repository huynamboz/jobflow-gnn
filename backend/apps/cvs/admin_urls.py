"""Admin-only CV management URLs (/api/admin/cvs/)."""

from django.urls import path

from apps.cvs.admin_views import (
    AdminCVDetailView,
    AdminCVListView,
    CVBatchListView,
    CVBatchDetailView,
    CVBatchRecordDetailView,
    CVBatchCancelView,
    CVExportView,
)

urlpatterns = [
    path("cvs/", AdminCVListView.as_view(), name="admin-cv-list"),
    path("cvs/export/", CVExportView.as_view(), name="admin-cv-export"),
    path("cvs/<int:pk>/", AdminCVDetailView.as_view(), name="admin-cv-detail"),
    path("cv/batches/", CVBatchListView.as_view(), name="admin-cv-batch-list"),
    path("cv/batches/<int:pk>/", CVBatchDetailView.as_view(), name="admin-cv-batch-detail"),
    path("cv/batches/<int:pk>/records/<int:rec_pk>/", CVBatchRecordDetailView.as_view(), name="admin-cv-batch-record-detail"),
    path("cv/batches/<int:pk>/cancel/", CVBatchCancelView.as_view(), name="admin-cv-batch-cancel"),
]
