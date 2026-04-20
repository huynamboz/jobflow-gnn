from django.urls import path

from .views import (
    LabelingExportView,
    LabelingQueueView,
    LabelingStatsView,
    SkipPairView,
    SubmitLabelView,
)

urlpatterns = [
    path("queue/",            LabelingQueueView.as_view(),  name="labeling-queue"),
    path("stats/",            LabelingStatsView.as_view(),  name="labeling-stats"),
    path("export/",           LabelingExportView.as_view(), name="labeling-export"),
    path("<int:pair_id>/submit/", SubmitLabelView.as_view(),   name="labeling-submit"),
    path("<int:pair_id>/skip/",   SkipPairView.as_view(),      name="labeling-skip"),
]
