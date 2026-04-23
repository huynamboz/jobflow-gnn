"""Admin-only CV management endpoints (/api/admin/cvs/)."""

import json

from django.db.models import Count
from django.http import HttpResponse
from drf_spectacular.utils import extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.cvs.models import CV, CVExtractionBatch, CVExtractionRecord, CVSkill
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
        role_category = request.query_params.get("role_category")
        if role_category:
            qs = qs.filter(role_category=role_category)

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


# --- CV Extraction Batch ---

class CVBatchListView(APIView):
    """GET /api/admin/cv/batches/ — list batches | POST — create + start new batch."""

    permission_classes = [IsAdmin]

    def get(self, request):
        batches = CVExtractionBatch.objects.all()[:50]
        return Response({
            "success": True,
            "data": [_serialize_batch(b) for b in batches],
        })

    def post(self, request):
        source = (request.data.get("source") or "").strip()
        source_categories = request.data.get("source_categories") or []  # list of strings
        cv_ids = request.data.get("cv_ids") or []                        # explicit list

        qs = CV.objects.filter(is_active=True)
        if cv_ids:
            qs = qs.filter(id__in=cv_ids)
        else:
            if source:
                qs = qs.filter(source=source)
            if source_categories:
                qs = qs.filter(source_category__in=source_categories)

        cv_list = list(qs.values_list("id", flat=True))
        if not cv_list:
            return Response(
                {"success": False, "error": {"code": "NO_CVS", "message": "No CVs match the filter.", "status": 400}},
                status=400,
            )

        batch = CVExtractionBatch.objects.create(
            filter_source=source,
            filter_source_categories=source_categories,
            total=len(cv_list),
        )
        CVExtractionRecord.objects.bulk_create([
            CVExtractionRecord(batch=batch, cv_id=cv_id)
            for cv_id in cv_list
        ])

        from apps.cvs.services.cv_batch_processor import start_batch
        start_batch(batch.id)

        return Response({"success": True, "data": _serialize_batch(batch)}, status=201)


class CVBatchDetailView(APIView):
    """GET /api/admin/cv/batches/:id/ — batch detail with record list."""

    permission_classes = [IsAdmin]

    def get(self, request, pk):
        try:
            batch = CVExtractionBatch.objects.get(pk=pk)
        except CVExtractionBatch.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Batch not found.", "status": 404}},
                status=404,
            )

        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 50))
        status_filter = request.query_params.get("status")

        records_qs = CVExtractionRecord.objects.filter(batch=batch).select_related("cv")
        if status_filter:
            records_qs = records_qs.filter(status=status_filter)

        total_records = records_qs.count()
        offset = (page - 1) * page_size
        records = records_qs[offset:offset + page_size]

        return Response({
            "success": True,
            "batch": _serialize_batch(batch),
            "records": [_serialize_record(r) for r in records],
            "total_records": total_records,
            "page": page,
            "page_size": page_size,
        })


class CVBatchRecordDetailView(APIView):
    """GET /api/admin/cv/batches/:pk/records/:rec_pk/ — raw_text + full extraction result."""

    permission_classes = [IsAdmin]

    def get(self, request, pk, rec_pk):
        try:
            record = CVExtractionRecord.objects.select_related("cv").get(pk=rec_pk, batch_id=pk)
        except CVExtractionRecord.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Record not found.", "status": 404}},
                status=404,
            )
        return Response({
            "success": True,
            "data": {
                "id": record.id,
                "cv_id": record.cv_id,
                "file_name": record.cv.file_name,
                "source_category": record.cv.source_category,
                "status": record.status,
                "error_msg": record.error_msg,
                "raw_text": record.cv.raw_text,
                "result": record.result,
            },
        })


class CVBatchCancelView(APIView):
    """POST /api/admin/cv/batches/:id/cancel/"""

    permission_classes = [IsAdmin]

    def post(self, request, pk):
        from apps.cvs.services.cv_batch_processor import cancel_batch
        cancelled = cancel_batch(pk)
        return Response({"success": True, "cancelled": cancelled})


def _serialize_batch(b: CVExtractionBatch) -> dict:
    return {
        "id": b.id,
        "filter_source": b.filter_source,
        "filter_source_categories": b.filter_source_categories,
        "status": b.status,
        "total": b.total,
        "done_count": b.done_count,
        "error_count": b.error_count,
        "created_at": b.created_at.isoformat(),
    }


def _serialize_record(r: CVExtractionRecord) -> dict:
    result = r.result or {}
    return {
        "id": r.id,
        "cv_id": r.cv_id,
        "file_name": r.cv.file_name,
        "source_category": r.cv.source_category,
        "status": r.status,
        "error_msg": r.error_msg,
        "role_category": result.get("role_category"),
        "seniority": result.get("seniority"),
        "experience_years": result.get("experience_years"),
        "skill_count": len(result.get("skills") or []),
    }


class CVExportView(APIView):
    """GET /api/admin/cvs/export/ — Download extracted CV data as JSON."""
    permission_classes = [IsAdmin]

    @extend_schema(tags=["Admin"])
    def get(self, request):
        role_filter = request.query_params.get("role_category", "")
        source_filter = request.query_params.get("source", "")

        qs = CV.objects.filter(is_active=True).order_by("id")
        if role_filter:
            qs = qs.filter(role_category=role_filter)
        if source_filter:
            qs = qs.filter(source=source_filter)

        # Fetch all skills in one query
        skill_map: dict[int, list[dict]] = {}
        for cs in (
            CVSkill.objects
            .filter(cv__in=qs)
            .select_related("skill")
            .values("cv_id", "skill__canonical_name", "proficiency")
        ):
            skill_map.setdefault(cs["cv_id"], []).append({
                "name": cs["skill__canonical_name"],
                "proficiency": cs["proficiency"],
            })

        records = []
        for cv in qs:
            records.append({
                "cv_id":          cv.id,
                "role_category":  cv.role_category or "other",
                "seniority":      cv.seniority,
                "experience_years": cv.experience_years,
                "education":      cv.education,
                "source":         cv.source,
                "source_category": cv.source_category,
                "skills":         skill_map.get(cv.id, []),
                "work_experience": cv.work_experience or [],
            })

        payload = json.dumps(records, ensure_ascii=False, indent=2)
        return HttpResponse(
            payload,
            content_type="application/json; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="cvs_extracted.json"'},
        )
