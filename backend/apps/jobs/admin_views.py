"""Admin-only job management endpoints (/api/admin/...)."""

from django.db.models import Count
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.jobs.models import Company, Job, Platform
from apps.jobs.serializers import (
    CompanySerializer,
    JobDetailSerializer,
    JobListSerializer,
    PlatformSerializer,
)
from apps.users.permissions import IsAdmin


# --- Platforms ---

class AdminPlatformListView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(responses={200: PlatformSerializer(many=True)}, tags=["Admin"])
    def get(self, request):
        platforms = Platform.objects.annotate(job_count=Count("jobs")).order_by("name")
        return Response({"success": True, "data": PlatformSerializer(platforms, many=True).data})


class AdminPlatformDetailView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(responses={200: PlatformSerializer}, tags=["Admin"])
    def get(self, request, pk):
        try:
            platform = Platform.objects.annotate(job_count=Count("jobs")).get(pk=pk)
        except Platform.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Platform not found.", "status": 404}}, status=404)
        return Response({"success": True, "data": PlatformSerializer(platform).data})

    @extend_schema(request=PlatformSerializer, responses={200: PlatformSerializer}, tags=["Admin"])
    def patch(self, request, pk):
        try:
            platform = Platform.objects.get(pk=pk)
        except Platform.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Platform not found.", "status": 404}}, status=404)
        serializer = PlatformSerializer(platform, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({"success": True, "data": serializer.data})


# --- Companies ---

class AdminCompanyListView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(responses={200: CompanySerializer(many=True)}, tags=["Admin"])
    def get(self, request):
        qs = Company.objects.annotate(job_count=Count("jobs")).prefetch_related("platforms")
        platform = request.query_params.get("platform")
        search = request.query_params.get("search")
        if platform:
            qs = qs.filter(platform_profiles__platform__slug=platform)
        if search:
            qs = qs.filter(name__icontains=search)
        return Response({"success": True, "data": CompanySerializer(qs[:100], many=True).data})


class AdminCompanyDetailView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(responses={200: CompanySerializer}, tags=["Admin"])
    def get(self, request, pk):
        try:
            company = Company.objects.annotate(job_count=Count("jobs")).prefetch_related("platforms").get(pk=pk)
        except Company.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Company not found.", "status": 404}}, status=404)
        return Response({"success": True, "data": CompanySerializer(company).data})


# --- Jobs ---

class AdminJobListView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(responses={200: JobListSerializer(many=True)}, tags=["Admin"])
    def get(self, request):
        qs = Job.objects.select_related("company", "platform")
        # Filters
        platform = request.query_params.get("platform")
        company_id = request.query_params.get("company")
        seniority = request.query_params.get("seniority")
        search = request.query_params.get("search")
        is_active = request.query_params.get("is_active")

        if platform:
            qs = qs.filter(platform__slug=platform)
        if company_id:
            qs = qs.filter(company_id=company_id)
        if seniority is not None:
            qs = qs.filter(seniority=seniority)
        if search:
            qs = qs.filter(title__icontains=search)
        if is_active is not None:
            qs = qs.filter(is_active=is_active.lower() == "true")

        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 20))
        offset = (page - 1) * page_size
        total = qs.count()

        return Response({
            "success": True,
            "data": JobListSerializer(qs[offset:offset + page_size], many=True).data,
            "total": total,
            "page": page,
            "page_size": page_size,
        })


class AdminJobDetailView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(responses={200: JobDetailSerializer}, tags=["Admin"])
    def get(self, request, pk):
        try:
            job = Job.objects.select_related("company", "platform").prefetch_related("job_skills__skill").get(pk=pk)
        except Job.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Job not found.", "status": 404}}, status=404)
        return Response({"success": True, "data": JobDetailSerializer(job).data})

    @extend_schema(tags=["Admin"])
    def patch(self, request, pk):
        try:
            job = Job.objects.get(pk=pk)
        except Job.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Job not found.", "status": 404}}, status=404)

        # Allow updating is_active, seniority, job_type
        for field in ("is_active", "seniority", "job_type"):
            if field in request.data:
                setattr(job, field, request.data[field])
        job.save()
        return Response({"success": True, "data": JobListSerializer(job).data})

    @extend_schema(tags=["Admin"])
    def delete(self, request, pk):
        try:
            job = Job.objects.get(pk=pk)
        except Job.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Job not found.", "status": 404}}, status=404)
        job.is_active = False
        job.save()
        return Response({"success": True, "message": "Job deactivated."})
