"""Admin-only user management endpoints (prefix: /api/admin/users/)."""

from django.contrib.auth import get_user_model
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.users.permissions import IsAdmin
from apps.users.serializers import UserSerializer

User = get_user_model()


class AdminUserListView(APIView):
    """GET /api/admin/users/ — List all users (admin only)."""

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: UserSerializer(many=True)})
    def get(self, request):
        users = User.objects.all().order_by("-date_joined")
        return Response({"success": True, "data": UserSerializer(users, many=True).data})


class AdminUserDetailView(APIView):
    """GET/PATCH/DELETE /api/admin/users/<id>/ — Manage single user (admin only)."""

    permission_classes = [IsAdmin]

    def _get_user(self, pk):
        try:
            return User.objects.get(pk=pk)
        except User.DoesNotExist:
            return None

    @extend_schema(responses={200: UserSerializer})
    def get(self, request, pk):
        user = self._get_user(pk)
        if not user:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "User not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response({"success": True, "data": UserSerializer(user).data})

    @extend_schema(request=UserSerializer, responses={200: UserSerializer})
    def patch(self, request, pk):
        user = self._get_user(pk)
        if not user:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "User not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )
        serializer = UserSerializer(user, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({"success": True, "data": serializer.data})

    def delete(self, request, pk):
        user = self._get_user(pk)
        if not user:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "User not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )
        user.is_active = False
        user.save()
        return Response({"success": True, "message": "User deactivated."})
