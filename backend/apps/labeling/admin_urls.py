from django.urls import path
from .admin_views import (
    LabelingBatchListView,
    LabelingBatchDetailView,
    LabelingBatchCancelView,
    LabelingBatchResumeView,
)

urlpatterns = [
    path("labeling/batches/",                    LabelingBatchListView.as_view(),   name="admin-labeling-batches"),
    path("labeling/batches/<int:pk>/",           LabelingBatchDetailView.as_view(), name="admin-labeling-batch-detail"),
    path("labeling/batches/<int:pk>/cancel/",    LabelingBatchCancelView.as_view(), name="admin-labeling-batch-cancel"),
    path("labeling/batches/<int:pk>/resume/",    LabelingBatchResumeView.as_view(), name="admin-labeling-batch-resume"),
]
