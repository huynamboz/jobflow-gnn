from django.urls import path

from apps.llm.views import (
    LLMCallLogDetailView,
    LLMCallLogListView,
    LLMProviderActivateView,
    LLMProviderDetailView,
    LLMProviderListCreateView,
    LLMProviderTestView,
)

urlpatterns = [
    path("llm/providers/", LLMProviderListCreateView.as_view(), name="llm-provider-list"),
    path("llm/providers/<int:pk>/", LLMProviderDetailView.as_view(), name="llm-provider-detail"),
    path("llm/providers/<int:pk>/activate/", LLMProviderActivateView.as_view(), name="llm-provider-activate"),
    path("llm/providers/<int:pk>/test/", LLMProviderTestView.as_view(), name="llm-provider-test"),
    path("llm/logs/", LLMCallLogListView.as_view(), name="llm-log-list"),
    path("llm/logs/<int:pk>/", LLMCallLogDetailView.as_view(), name="llm-log-detail"),
]
