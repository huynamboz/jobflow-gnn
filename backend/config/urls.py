from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    # Django Admin (built-in)
    path("admin/", admin.site.urls),

    # Auth (public)
    path("api/auth/", include("apps.users.urls")),

    # User endpoints (authenticated)
    path("api/matching/", include("apps.matching.urls")),

    # Admin-only management endpoints
    path("api/admin/users/", include("apps.users.admin_urls")),

    # Swagger
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger"),
]
