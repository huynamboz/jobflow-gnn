from django.urls import path

from apps.cvs.views import ExtractCVView, MyCVDetailView, MyCVListView, SaveCVView, UploadCVView

urlpatterns = [
    path("", MyCVListView.as_view(), name="cv-list"),
    path("upload/", UploadCVView.as_view(), name="cv-upload"),
    path("extract/", ExtractCVView.as_view(), name="cv-extract"),
    path("save/", SaveCVView.as_view(), name="cv-save"),
    path("<int:pk>/", MyCVDetailView.as_view(), name="cv-detail"),
]
