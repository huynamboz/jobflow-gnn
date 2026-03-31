"""User-facing CV endpoints (/api/cvs/)."""

import tempfile
from pathlib import Path

from drf_spectacular.utils import extend_schema, inline_serializer
from rest_framework import serializers, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.cvs.models import CV
from apps.cvs.serializers import CVDetailSerializer, CVListSerializer
from apps.cvs.services import CVService


class MyCVListView(APIView):
    """GET /api/cvs/ — List my uploaded CVs."""

    permission_classes = [IsAuthenticated]

    @extend_schema(responses={200: CVListSerializer(many=True)}, tags=["CVs"])
    def get(self, request):
        cvs = CV.objects.filter(user=request.user, is_active=True).annotate(
            skill_count=models.Count("cv_skills")
        )
        return Response({"success": True, "data": CVListSerializer(cvs, many=True).data})


class MyCVDetailView(APIView):
    """GET/DELETE /api/cvs/<id>/ — View or delete my CV."""

    permission_classes = [IsAuthenticated]

    @extend_schema(responses={200: CVDetailSerializer}, tags=["CVs"])
    def get(self, request, pk):
        try:
            cv = CV.objects.filter(user=request.user).prefetch_related("cv_skills__skill").get(pk=pk)
        except CV.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "CV not found.", "status": 404}},
                status=404,
            )
        return Response({"success": True, "data": CVDetailSerializer(cv).data})

    def delete(self, request, pk):
        try:
            cv = CV.objects.filter(user=request.user).get(pk=pk)
        except CV.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "CV not found.", "status": 404}},
                status=404,
            )
        cv.is_active = False
        cv.save()
        return Response({"success": True, "message": "CV deleted."})


class UploadCVView(APIView):
    """POST /api/cvs/upload/ — Upload CV file, parse, save to DB."""

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "format": "binary", "description": "CV file (PDF/DOCX/TXT)"},
                },
                "required": ["file"],
            }
        },
        responses={201: CVDetailSerializer},
        tags=["CVs"],
    )
    def post(self, request):
        file = request.FILES.get("file")
        if not file:
            return Response(
                {"success": False, "error": {"code": "BAD_REQUEST", "message": "No file uploaded.", "status": 400}},
                status=400,
            )

        suffix = Path(file.name).suffix.lower()
        if suffix not in (".pdf", ".docx", ".txt"):
            return Response(
                {"success": False, "error": {"code": "BAD_REQUEST", "message": f"Unsupported file type: {suffix}.", "status": 400}},
                status=400,
            )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            for chunk in file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            service = CVService()
            cv = service.save_from_file(tmp_path, user=request.user, source="upload")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not cv.cv_skills.exists():
            return Response(
                {"success": False, "error": {"code": "UNPROCESSABLE_ENTITY", "message": "No skills extracted from CV.", "status": 422}},
                status=422,
            )

        cv_detail = CV.objects.prefetch_related("cv_skills__skill").get(pk=cv.pk)
        return Response(
            {"success": True, "data": CVDetailSerializer(cv_detail).data},
            status=status.HTTP_201_CREATED,
        )


# Fix import for annotation
from django.db import models
