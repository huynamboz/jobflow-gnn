"""Admin-only API views for LLM labeling batch management."""

from django.db.models import Count
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.users.permissions import IsAdmin
from .models import LabelingBatch, PairQueue, PairStatus, HumanLabel, SelectionReason
from .services.label_batch_processor import start_batch, cancel_batch, is_running, WORKERS


def _batch_to_dict(batch: LabelingBatch) -> dict:
    pct = round(batch.done_count / batch.total * 100, 1) if batch.total > 0 else 0
    return {
        "id":          batch.id,
        "status":      batch.status,
        "total":       batch.total,
        "done_count":  batch.done_count,
        "error_count": batch.error_count,
        "workers":     batch.workers,
        "pct":         pct,
        "created_at":  batch.created_at.isoformat(),
    }


class LabelingBatchListView(APIView):
    permission_classes = [IsAdmin]

    def get(self, request):
        """List all labeling batches + queue stats."""
        batches = list(LabelingBatch.objects.all()[:20])
        pending  = PairQueue.objects.filter(status=PairStatus.PENDING).count()
        labeled  = PairQueue.objects.filter(status=PairStatus.LABELED).count()
        total_q  = PairQueue.objects.count()
        label_count = HumanLabel.objects.count()

        overall_dist = {0: 0, 1: 0, 2: 0}
        for row in HumanLabel.objects.values("overall").annotate(cnt=Count("id")):
            overall_dist[row["overall"]] = row["cnt"]

        return Response({
            "success": True,
            "data": {
                "batches": [_batch_to_dict(b) for b in batches],
                "queue": {
                    "total":   total_q,
                    "pending": pending,
                    "labeled": labeled,
                    "skipped": total_q - pending - labeled,
                },
                "labels": {
                    "total": label_count,
                    "overall_0": overall_dist[0],
                    "overall_1": overall_dist[1],
                    "overall_2": overall_dist[2],
                },
            },
        })

    def post(self, request):
        """Start a new labeling batch."""
        workers_raw = request.data.get("workers")
        try:
            workers = max(1, min(20, int(workers_raw))) if workers_raw not in (None, "", "null") else WORKERS
        except (TypeError, ValueError):
            workers = WORKERS

        pending = PairQueue.objects.filter(status=PairStatus.PENDING).count()
        if pending == 0:
            return Response(
                {"success": False, "error": {"code": "NO_PENDING", "message": "No pending pairs to label.", "status": 400}},
                status=400,
            )

        # Don't allow starting if one is already running
        running_batch = LabelingBatch.objects.filter(status=LabelingBatch.STATUS_RUNNING).first()
        if running_batch and is_running(running_batch.id):
            return Response(
                {"success": False, "error": {"code": "ALREADY_RUNNING", "message": f"Batch #{running_batch.id} is already running.", "status": 400}},
                status=400,
            )

        batch = LabelingBatch.objects.create(workers=workers)
        start_batch(batch.id, workers=workers)
        return Response({"success": True, "data": _batch_to_dict(batch)})


class LabelingBatchDetailView(APIView):
    permission_classes = [IsAdmin]

    def get(self, request, pk):
        try:
            batch = LabelingBatch.objects.get(pk=pk)
        except LabelingBatch.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "status": 404}}, status=404)

        recent = list(
            HumanLabel.objects
            .filter(batch_id=pk)
            .select_related("pair__cv", "pair__job")
            .order_by("-created_at")[:50]
        )
        def _top_skills(skills: list, key: str, n: int = 8) -> list[str]:
            if not skills:
                return []
            if isinstance(skills[0], dict):
                return [s["name"] for s in sorted(skills, key=lambda s: -s.get(key, 0))[:n]]
            return [str(s) for s in skills[:n]]

        recent_data = []
        for lbl in recent:
            cv  = lbl.pair.cv
            job = lbl.pair.job
            recent_data.append({
                "pair_id":        lbl.pair_id,
                "cv_id":          cv.cv_id,
                "job_id":         job.job_id,
                "job_title":      job.title,
                "cv_role":        cv.role_category,
                "cv_seniority":   cv.seniority,
                "cv_experience":  cv.experience_years,
                "cv_education":   cv.education,
                "cv_skills":      _top_skills(cv.skills or [], "proficiency"),
                "cv_text":        (cv.text_summary or "")[:500],
                "job_role":       job.role_category,
                "job_seniority":  job.seniority,
                "job_experience": f"{job.experience_min:.0f}–{job.experience_max:.0f}y" if job.experience_max else (f"≥{job.experience_min:.0f}y" if job.experience_min else "—"),
                "job_skills":     _top_skills(job.skills or [], "importance"),
                "job_text":       (job.text_summary or "")[:500],
                "skill_fit":      lbl.skill_fit,
                "seniority_fit":  lbl.seniority_fit,
                "experience_fit": lbl.experience_fit,
                "domain_fit":     lbl.domain_fit,
                "overall":        lbl.overall,
                "selection":      lbl.pair.selection_reason,
                "created_at":     lbl.created_at.isoformat(),
            })

        # Score distribution for this batch
        dist = {0: 0, 1: 0, 2: 0}
        for row in HumanLabel.objects.filter(batch_id=pk).values("overall").annotate(cnt=Count("id")):
            dist[row["overall"]] = row["cnt"]

        return Response({
            "success": True,
            "data": {
                **_batch_to_dict(batch),
                "recent": recent_data,
                "dist": {"overall_0": dist[0], "overall_1": dist[1], "overall_2": dist[2]},
            },
        })


class LabelingBatchCancelView(APIView):
    permission_classes = [IsAdmin]

    def post(self, request, pk):
        try:
            batch = LabelingBatch.objects.get(pk=pk)
        except LabelingBatch.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "status": 404}}, status=404)

        if batch.status != LabelingBatch.STATUS_RUNNING:
            return Response(
                {"success": False, "error": {"code": "NOT_RUNNING", "message": "Batch is not running.", "status": 400}},
                status=400,
            )

        cancel_batch(batch.id)
        return Response({"success": True})


class LabelingBatchResumeView(APIView):
    permission_classes = [IsAdmin]

    def post(self, request, pk):
        try:
            batch = LabelingBatch.objects.get(pk=pk)
        except LabelingBatch.DoesNotExist:
            return Response({"success": False, "error": {"code": "NOT_FOUND", "status": 404}}, status=404)

        if batch.status not in (LabelingBatch.STATUS_ERROR, LabelingBatch.STATUS_CANCELLED):
            return Response(
                {"success": False, "error": {"code": "NOT_RESUMABLE", "message": "Only error or cancelled batches can be resumed.", "status": 400}},
                status=400,
            )

        pending = PairQueue.objects.filter(status=PairStatus.PENDING).count()
        if pending == 0:
            return Response(
                {"success": False, "error": {"code": "NO_PENDING", "message": "No pending pairs to label.", "status": 400}},
                status=400,
            )

        workers_raw = request.data.get("workers")
        try:
            workers = max(1, min(20, int(workers_raw))) if workers_raw not in (None, "", "null") else batch.workers
        except (TypeError, ValueError):
            workers = batch.workers

        LabelingBatch.objects.filter(id=pk).update(status=LabelingBatch.STATUS_RUNNING, workers=workers)
        start_batch(batch.id, workers=workers)
        batch.refresh_from_db()
        return Response({"success": True, "data": _batch_to_dict(batch)})
