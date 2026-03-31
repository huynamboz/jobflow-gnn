"""User-facing job endpoints (/api/jobs/)."""

from drf_spectacular.utils import extend_schema
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.jobs.models import Job
from apps.jobs.serializers import JobDetailSerializer, JobListSerializer


class JobListView(APIView):
    """GET /api/jobs/ — Search and filter jobs (public)."""

    permission_classes = [AllowAny]

    @extend_schema(responses={200: JobListSerializer(many=True)}, tags=["Jobs"])
    def get(self, request):
        qs = Job.objects.filter(is_active=True).select_related("company", "platform")

        search = request.query_params.get("search")
        platform = request.query_params.get("platform")
        company = request.query_params.get("company")
        seniority = request.query_params.get("seniority")
        job_type = request.query_params.get("job_type")
        location = request.query_params.get("location")
        skill = request.query_params.get("skill")

        if search:
            qs = qs.filter(title__icontains=search)
        if platform:
            qs = qs.filter(platform__slug=platform)
        if company:
            qs = qs.filter(company_id=company)
        if seniority is not None and seniority != "":
            qs = qs.filter(seniority=seniority)
        if job_type:
            qs = qs.filter(job_type=job_type)
        if location:
            qs = qs.filter(location__icontains=location)
        if skill:
            qs = qs.filter(job_skills__skill__canonical_name=skill)

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


class JobDetailView(APIView):
    """GET /api/jobs/<id>/ — Job detail with skills (public)."""

    permission_classes = [AllowAny]

    @extend_schema(responses={200: JobDetailSerializer}, tags=["Jobs"])
    def get(self, request, pk):
        try:
            job = Job.objects.filter(is_active=True).select_related(
                "company", "platform"
            ).prefetch_related("job_skills__skill").get(pk=pk)
        except Job.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Job not found.", "status": 404}},
                status=404,
            )
        return Response({"success": True, "data": JobDetailSerializer(job).data})
