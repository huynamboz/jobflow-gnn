"""Admin-only CV management endpoints (/api/admin/cvs/)."""

from django.db.models import Count
from drf_spectacular.utils import extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.cvs.models import CV
from apps.cvs.serializers import CVDetailSerializer, CVListSerializer
from apps.users.permissions import IsAdmin


class AdminCVListView(APIView):
    """GET /api/admin/cvs/ — List all CVs in the system (admin only)."""

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: CVListSerializer(many=True)}, tags=["Admin"])
    def get(self, request):
        qs = CV.objects.filter(is_active=True).annotate(skill_count=Count("cv_skills"))

        source = request.query_params.get("source")
        seniority = request.query_params.get("seniority")
        search = request.query_params.get("search")

        if source:
            qs = qs.filter(source=source)
        if seniority is not None and seniority != "":
            qs = qs.filter(seniority=seniority)
        if search:
            qs = qs.filter(file_name__icontains=search)

        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 20))
        offset = (page - 1) * page_size
        total = qs.count()

        return Response({
            "success": True,
            "data": CVListSerializer(qs.order_by("-created_at")[offset:offset + page_size], many=True).data,
            "total": total,
            "page": page,
            "page_size": page_size,
        })


class AdminCVDetailView(APIView):
    """GET /api/admin/cvs/<id>/ — CV detail with full skills list."""

    permission_classes = [IsAdmin]

    @extend_schema(responses={200: CVDetailSerializer}, tags=["Admin"])
    def get(self, request, pk):
        try:
            cv = CV.objects.prefetch_related("cv_skills__skill").get(pk=pk, is_active=True)
        except CV.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "CV not found.", "status": 404}},
                status=404,
            )
        return Response({"success": True, "data": CVDetailSerializer(cv).data})
