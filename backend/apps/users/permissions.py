from rest_framework.permissions import BasePermission


class IsAdmin(BasePermission):
    """Only users with role='admin' or is_staff=True."""

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and (request.user.role == "admin" or request.user.is_staff)
        )


class IsRecruiterOrAdmin(BasePermission):
    """Users with role='recruiter' or 'admin'."""

    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.user.role in ("recruiter", "admin")
        )
