from django.urls import path

from apps.cvs.views import MyCVDetailView, MyCVListView, UploadCVView

urlpatterns = [
    path("", MyCVListView.as_view(), name="cv-list"),
    path("upload/", UploadCVView.as_view(), name="cv-upload"),
    path("<int:pk>/", MyCVDetailView.as_view(), name="cv-detail"),
]
