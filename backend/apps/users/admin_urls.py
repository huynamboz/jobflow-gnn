"""Admin-only user management URLs (prefix: /api/admin/users/)."""

from django.urls import path

from apps.users.admin_views import AdminUserDetailView, AdminUserListView

urlpatterns = [
    path("", AdminUserListView.as_view(), name="admin-user-list"),
    path("<int:pk>/", AdminUserDetailView.as_view(), name="admin-user-detail"),
]
