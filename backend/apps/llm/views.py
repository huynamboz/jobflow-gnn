"""Admin views for LLM provider management."""

from django.db import transaction
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.llm.models import LLMCallLog, LLMProvider
from apps.llm.serializers import LLMCallLogSerializer, LLMProviderSerializer, LLMProviderTestSerializer, LLMProviderWriteSerializer
from apps.llm.service import _build_client
from apps.users.permissions import IsAdmin


class LLMProviderListCreateView(APIView):
    """GET /api/admin/llm/providers/  — list all providers.
    POST /api/admin/llm/providers/ — create a new provider.
    """

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: LLMProviderSerializer(many=True)}, tags=["Admin - LLM"])
    def get(self, request):
        providers = LLMProvider.objects.all()
        return Response({"success": True, "data": LLMProviderSerializer(providers, many=True).data})

    @extend_schema(request=LLMProviderWriteSerializer, responses={201: LLMProviderSerializer}, tags=["Admin - LLM"])
    def post(self, request):
        ser = LLMProviderWriteSerializer(data=request.data)
        if not ser.is_valid():
            return Response(
                {"success": False, "error": {"code": "VALIDATION_ERROR", "message": ser.errors, "status": 400}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        provider = ser.save()
        return Response(
            {"success": True, "data": LLMProviderSerializer(provider).data},
            status=status.HTTP_201_CREATED,
        )


class LLMProviderDetailView(APIView):
    """GET/PUT/DELETE /api/admin/llm/providers/{id}/"""

    permission_classes = [IsAdmin]

    def _get_object(self, pk: int):
        try:
            return LLMProvider.objects.get(pk=pk)
        except LLMProvider.DoesNotExist:
            return None

    @extend_schema(responses={200: LLMProviderSerializer}, tags=["Admin - LLM"])
    def get(self, request, pk: int):
        provider = self._get_object(pk)
        if not provider:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Provider not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response({"success": True, "data": LLMProviderSerializer(provider).data})

    @extend_schema(request=LLMProviderWriteSerializer, responses={200: LLMProviderSerializer}, tags=["Admin - LLM"])
    def put(self, request, pk: int):
        provider = self._get_object(pk)
        if not provider:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Provider not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )
        ser = LLMProviderWriteSerializer(provider, data=request.data, partial=True)
        if not ser.is_valid():
            return Response(
                {"success": False, "error": {"code": "VALIDATION_ERROR", "message": ser.errors, "status": 400}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        provider = ser.save()
        return Response({"success": True, "data": LLMProviderSerializer(provider).data})

    @extend_schema(tags=["Admin - LLM"])
    def delete(self, request, pk: int):
        provider = self._get_object(pk)
        if not provider:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Provider not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )
        if provider.is_active:
            return Response(
                {"success": False, "error": {"code": "CONFLICT", "message": "Cannot delete the active provider. Activate another first.", "status": 409}},
                status=status.HTTP_409_CONFLICT,
            )
        provider.delete()
        return Response({"success": True, "data": None}, status=status.HTTP_200_OK)


class LLMProviderActivateView(APIView):
    """POST /api/admin/llm/providers/{id}/activate/ — set as the single active provider."""

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: LLMProviderSerializer}, tags=["Admin - LLM"])
    def post(self, request, pk: int):
        try:
            provider = LLMProvider.objects.get(pk=pk)
        except LLMProvider.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Provider not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )

        with transaction.atomic():
            LLMProvider.objects.exclude(pk=pk).update(is_active=False)
            provider.is_active = True
            provider.save(update_fields=["is_active", "updated_at"])

        return Response({"success": True, "data": LLMProviderSerializer(provider).data})


class LLMProviderTestView(APIView):
    """POST /api/admin/llm/providers/{id}/test/ — send a test message to verify credentials."""

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: LLMProviderTestSerializer}, tags=["Admin - LLM"])
    def post(self, request, pk: int):
        try:
            provider = LLMProvider.objects.get(pk=pk)
        except LLMProvider.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Provider not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )

        client = _build_client(provider)
        ok, message = client.test_connection()

        return Response({"success": True, "data": {"ok": ok, "message": message}})


class LLMCallLogListView(APIView):
    """GET /api/admin/llm/logs/ — list call logs with pagination."""

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: LLMCallLogSerializer(many=True)}, tags=["Admin - LLM"])
    def get(self, request):
        qs = LLMCallLog.objects.select_related("provider")

        status_filter = request.query_params.get("status")
        feature_filter = request.query_params.get("feature")
        if status_filter:
            qs = qs.filter(status=status_filter)
        if feature_filter:
            qs = qs.filter(feature=feature_filter)

        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 50))
        total = qs.count()
        offset = (page - 1) * page_size

        return Response({
            "success": True,
            "data": LLMCallLogSerializer(qs[offset:offset + page_size], many=True).data,
            "total": total,
            "page": page,
            "page_size": page_size,
        })


class LLMCallLogDetailView(APIView):
    """GET /api/admin/llm/logs/<id>/ — full log detail."""

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: LLMCallLogSerializer}, tags=["Admin - LLM"])
    def get(self, request, pk: int):
        try:
            log = LLMCallLog.objects.select_related("provider").get(pk=pk)
        except LLMCallLog.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Log not found.", "status": 404}},
                status=404,
            )
        return Response({"success": True, "data": LLMCallLogSerializer(log).data})
