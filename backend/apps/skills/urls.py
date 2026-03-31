from django.urls import path

from apps.skills.views import SkillListView

urlpatterns = [
    path("", SkillListView.as_view(), name="skill-list"),
]
