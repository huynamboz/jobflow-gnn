"""Admin-only job management endpoints (/api/admin/...)."""

from django.db.models import Count
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.jobs.models import Company, Job, JobSkill, Platform
from apps.jobs.serializers import (
    CompanySerializer,
    JobDetailSerializer,
    JobListSerializer,
    PlatformSerializer,
)
from apps.users.permissions import IsAdmin


# --- JD Extraction ---

class JDExtractView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(tags=["Admin"])
    def post(self, request):
        raw_text = (request.data.get("raw_text") or "").strip()
        if not raw_text:
            return Response(
                {"success": False, "error": {"code": "INVALID_INPUT", "message": "raw_text is required.", "status": 400}},
                status=400,
            )

        from apps.jobs.services.llm_jd_extractor import extract
        result = extract(raw_text)

        return Response({
            "success": True,
            "data": {
                "title": result.title,
                "company": result.company,
                "location": result.location,
                "seniority": result.seniority,
                "job_type": result.job_type,
                "salary_min": result.salary_min,
                "salary_max": result.salary_max,
                "salary_currency": result.salary_currency,
                "salary_type": result.salary_type,
                "experience_min": result.experience_min,
                "experience_max": result.experience_max,
                "degree_requirement": result.degree_requirement,
                "skills": result.skills,
            },
        })


class JDSaveView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(tags=["Admin"])
    def post(self, request):
        data = request.data
        title = (data.get("title") or "").strip()
        if not title:
            return Response(
                {"success": False, "error": {"code": "INVALID_INPUT", "message": "title is required.", "status": 400}},
                status=400,
            )

        # Resolve or create company
        company = None
        company_name = (data.get("company") or "").strip()
        if company_name:
            company, _ = Company.objects.get_or_create(name=company_name)

        job = Job.objects.create(
            title=title,
            company=company,
            location=(data.get("location") or "").strip(),
            seniority=int(data.get("seniority") or 2),
            job_type=data.get("job_type") or Job.JobType.OTHER,
            salary_min=int(data.get("salary_min") or 0),
            salary_max=int(data.get("salary_max") or 0),
            salary_currency=(data.get("salary_currency") or "USD").upper(),
            description=data.get("raw_text") or "",
        )

        # Attach skills
        from apps.skills.services import SkillService
        skill_svc = SkillService()
        for s in (data.get("skills") or []):
            name = (s.get("name") or "").strip()
            if not name:
                continue
            skill = skill_svc.get_or_create(name)
            if skill:
                JobSkill.objects.get_or_create(
                    job=job, skill=skill,
                    defaults={"importance": int(s.get("importance") or 3)},
                )

        return Response({"success": True, "data": {"id": job.id}}, status=status.HTTP_201_CREATED)


# --- JD Batch Extraction ---

def _batch_to_dict(batch) -> dict:
    return {
        "id": batch.id,
        "file_path": batch.file_path,
        "fields_config": batch.fields_config,
        "status": batch.status,
        "total": batch.total,
        "done_count": batch.done_count,
        "error_count": batch.error_count,
        "created_at": batch.created_at.isoformat(),
    }


def _record_to_dict(record, include_result: bool = False) -> dict:
    d = {
        "id": record.id,
        "index": record.index,
        "status": record.status,
        "error_msg": record.error_msg,
        "title": (record.raw_data or {}).get("title", ""),
        "company": (record.raw_data or {}).get("company", ""),
    }
    if include_result:
        d["result"] = record.result
        d["combined_text"] = record.combined_text
    return d


def _parse_jsonl(file_obj, limit=None):
    """Read a JSONL file object, return (records list, fields set, total count)."""
    import json as _json
    records = []
    fields: set[str] = set()
    total = 0
    idx = 0
    for raw in file_obj:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line:
            continue
        try:
            obj = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        fields.update(obj.keys())
        total += 1
        if limit is None or idx < limit:
            records.append((idx, obj))
        idx += 1
    return records, fields, total


class JDBatchPreviewView(APIView):
    permission_classes = [IsAdmin]
    parser_classes = [__import__("rest_framework.parsers", fromlist=["MultiPartParser"]).MultiPartParser]

    @extend_schema(tags=["Admin"])
    def post(self, request):
        upload = request.FILES.get("file")
        if not upload:
            return Response({"success": False, "error": {"code": "INVALID_INPUT", "message": "file is required.", "status": 400}}, status=400)

        try:
            records, fields, total = _parse_jsonl(upload, limit=None)
        except Exception as exc:
            return Response({"success": False, "error": {"code": "READ_ERROR", "message": str(exc), "status": 400}}, status=400)

        sample = [obj for _, obj in records[:5]]
        return Response({
            "success": True,
            "data": {
                "total": total,
                "fields": sorted(fields),
                "sample": sample,
                "filename": upload.name,
            },
        })


class JDBatchListView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(tags=["Admin"])
    def get(self, request):
        from apps.jobs.models import JDExtractionBatch
        batches = JDExtractionBatch.objects.all()[:20]
        return Response({"success": True, "data": [_batch_to_dict(b) for b in batches]})

    @extend_schema(tags=["Admin"])
    def post(self, request):
        import json as _json
        from rest_framework.parsers import MultiPartParser, JSONParser
        from apps.jobs.models import JDExtractionBatch, JDExtractionRecord
        from apps.jobs.services.jd_batch_processor import start_batch, content_hash

        upload = request.FILES.get("file")
        # fields_config comes as JSON string in multipart or list in JSON body
        raw_fields = request.data.get("fields_config") or "[]"
        if isinstance(raw_fields, str):
            try:
                fields_config = _json.loads(raw_fields)
            except _json.JSONDecodeError:
                fields_config = []
        else:
            fields_config = list(raw_fields)

        limit_raw = request.data.get("limit")
        limit = int(limit_raw) if limit_raw not in (None, "", "null") else None

        if not upload:
            return Response({"success": False, "error": {"code": "INVALID_INPUT", "message": "file is required.", "status": 400}}, status=400)
        if not fields_config:
            return Response({"success": False, "error": {"code": "INVALID_INPUT", "message": "fields_config required.", "status": 400}}, status=400)

        batch = JDExtractionBatch.objects.create(
            file_path=upload.name,
            fields_config=fields_config,
        )

        records_to_create = []
        try:
            all_records, _, _ = _parse_jsonl(upload, limit=limit)
            for idx, obj in all_records:
                parts = [f"{f}: {obj[f]}" for f in fields_config if obj.get(f)]
                combined = "\n".join(parts)
                records_to_create.append(JDExtractionRecord(
                    batch=batch,
                    index=idx,
                    raw_data=obj,
                    combined_text=combined,
                    content_hash=content_hash(combined),
                ))
        except Exception as exc:
            batch.delete()
            return Response({"success": False, "error": {"code": "READ_ERROR", "message": str(exc), "status": 400}}, status=400)

        JDExtractionRecord.objects.bulk_create(records_to_create, batch_size=500)
        batch.total = len(records_to_create)
        batch.save(update_fields=["total"])

        start_batch(batch.id)

        return Response({"success": True, "data": _batch_to_dict(batch)}, status=status.HTTP_201_CREATED)


class JDBatchDetailView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(tags=["Admin"])
    def get(self, request, pk):
        from apps.jobs.models import JDExtractionBatch, JDExtractionRecord

        try:
            batch = JDExtractionBatch.objects.get(pk=pk)
        except JDExtractionBatch.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Batch not found.", "status": 404}}, status=404)

        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 50))
        status_filter = request.query_params.get("status", "")

        records_qs = JDExtractionRecord.objects.filter(batch=batch)
        if status_filter:
            records_qs = records_qs.filter(status=status_filter)

        total_records = records_qs.count()
        offset = (page - 1) * page_size
        records = records_qs[offset: offset + page_size]

        return Response({
            "success": True,
            "data": {
                "batch": _batch_to_dict(batch),
                "records": [_record_to_dict(r) for r in records],
                "total_records": total_records,
                "page": page,
                "page_size": page_size,
            },
        })


class JDBatchCancelView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(tags=["Admin"])
    def post(self, request, pk):
        from apps.jobs.models import JDExtractionBatch
        from apps.jobs.services.jd_batch_processor import cancel_batch

        try:
            batch = JDExtractionBatch.objects.get(pk=pk)
        except JDExtractionBatch.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Batch not found.", "status": 404}}, status=404)

        if batch.status != JDExtractionBatch.STATUS_RUNNING:
            return Response({"success": False, "error": {"code": "INVALID_STATE", "message": "Batch is not running.", "status": 400}}, status=400)

        cancel_batch(batch.id)
        return Response({"success": True})


class JDBatchRecordDetailView(APIView):
    permission_classes = [IsAdmin]

    @extend_schema(tags=["Admin"])
    def get(self, request, pk, record_pk):
        from apps.jobs.models import JDExtractionRecord

        try:
            record = JDExtractionRecord.objects.get(pk=record_pk, batch_id=pk)
        except JDExtractionRecord.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "message": "Record not found.", "status": 404}}, status=404)

        return Response({"success": True, "data": _record_to_dict(record, include_result=True)})


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
