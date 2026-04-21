import mimetypes
import os

from django.db.models import Count, Q
from django.http import FileResponse, Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.users.permissions import IsAdmin

from .models import HumanLabel, LabelingCV, PairQueue, PairStatus, SelectionReason
from .serializers import (
    ExportItemSerializer,
    LabelingCVSerializer,
    PairQueueItemSerializer,
    SubmitLabelSerializer,
)


class LabelingQueueView(APIView):
    """GET /api/labeling/queue/ — Return current CV and all its pending pairs."""

    permission_classes = [IsAdmin]

    def get(self, request):
        # Find first CV that has pending pairs, ordered by priority
        first_pending = (
            PairQueue.objects.filter(status=PairStatus.PENDING)
            .order_by("priority", "cv_id")
            .select_related("cv")
            .first()
        )

        if first_pending is None:
            return Response(
                {"success": True, "data": None, "message": "All pairs labeled!"},
                status=status.HTTP_200_OK,
            )

        current_cv = first_pending.cv

        # Get all pending pairs for this CV
        pending_pairs = (
            PairQueue.objects.filter(cv=current_cv, status=PairStatus.PENDING)
            .order_by("priority")
            .select_related("job")
        )

        # Progress stats
        total = PairQueue.objects.count()
        labeled = PairQueue.objects.filter(status=PairStatus.LABELED).count()
        skipped = PairQueue.objects.filter(status=PairStatus.SKIPPED).count()

        data = {
            "cv": LabelingCVSerializer(current_cv).data,
            "pairs": PairQueueItemSerializer(pending_pairs, many=True).data,
            "progress": {
                "total": total,
                "labeled": labeled,
                "skipped": skipped,
                "pending": total - labeled - skipped,
                "current_cv_pending": pending_pairs.count(),
            },
        }
        return Response({"success": True, "data": data})


class SubmitLabelView(APIView):
    """POST /api/labeling/{pair_id}/submit/ — Submit label for a pair."""

    permission_classes = [IsAdmin]

    def post(self, request, pair_id):
        try:
            pair = PairQueue.objects.get(pk=pair_id)
        except PairQueue.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Pair not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )

        if pair.status == PairStatus.LABELED:
            return Response(
                {"success": False, "error": {"code": "CONFLICT", "message": "Pair already labeled.", "status": 409}},
                status=status.HTTP_409_CONFLICT,
            )

        serializer = SubmitLabelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        HumanLabel.objects.create(
            pair=pair,
            skill_fit=data["skill_fit"],
            seniority_fit=data["seniority_fit"],
            experience_fit=data["experience_fit"],
            domain_fit=data["domain_fit"],
            overall=data["overall"],
            note=data.get("note", ""),
            labeled_by=request.user,
        )

        pair.status = PairStatus.LABELED
        pair.save(update_fields=["status"])

        return Response({"success": True, "message": "Label saved."})


class SkipPairView(APIView):
    """POST /api/labeling/{pair_id}/skip/ — Skip a pair."""

    permission_classes = [IsAdmin]

    def post(self, request, pair_id):
        try:
            pair = PairQueue.objects.get(pk=pair_id)
        except PairQueue.DoesNotExist:
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "Pair not found.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )

        pair.status = PairStatus.SKIPPED
        pair.save(update_fields=["status"])
        return Response({"success": True, "message": "Pair skipped."})


class LabelingStatsView(APIView):
    """GET /api/labeling/stats/ — Labeling progress and distribution."""

    permission_classes = [IsAdmin]

    def get(self, request):
        total = PairQueue.objects.count()
        labeled = PairQueue.objects.filter(status=PairStatus.LABELED).count()
        skipped = PairQueue.objects.filter(status=PairStatus.SKIPPED).count()

        # Label value distribution
        label_dist = {str(v): 0 for v in [0, 1, 2]}
        for hl in HumanLabel.objects.values("overall"):
            label_dist[str(hl["overall"])] = label_dist.get(str(hl["overall"]), 0) + 1

        # By selection reason
        by_reason = {}
        for reason in SelectionReason.values:
            reason_total = PairQueue.objects.filter(selection_reason=reason).count()
            reason_labeled = PairQueue.objects.filter(selection_reason=reason, status=PairStatus.LABELED).count()
            by_reason[reason] = {"labeled": reason_labeled, "total": reason_total}

        # By split
        by_split = {}
        for split in ["train", "val", "test"]:
            split_total = PairQueue.objects.filter(split=split).count()
            split_labeled = PairQueue.objects.filter(split=split, status=PairStatus.LABELED).count()
            by_split[split] = {"labeled": split_labeled, "total": split_total}

        data = {
            "total_pairs": total,
            "labeled": labeled,
            "skipped": skipped,
            "pending": total - labeled - skipped,
            "label_distribution": label_dist,
            "by_reason": by_reason,
            "by_split": by_split,
        }
        return Response({"success": True, "data": data})


class CVPdfView(APIView):
    """GET /api/labeling/cvs/<cv_id>/pdf/ — Serve the CV PDF file."""

    permission_classes = [IsAdmin]

    def get(self, request, cv_id):
        try:
            cv = LabelingCV.objects.get(cv_id=cv_id)
        except LabelingCV.DoesNotExist:
            raise Http404

        if not cv.pdf_path or not os.path.isfile(cv.pdf_path):
            return Response(
                {"success": False, "error": {"code": "NOT_FOUND", "message": "PDF not available.", "status": 404}},
                status=status.HTTP_404_NOT_FOUND,
            )

        return FileResponse(open(cv.pdf_path, "rb"), content_type="application/pdf")


class LabelingExportView(APIView):
    """GET /api/labeling/export/ — Export labeled pairs for ML training."""

    permission_classes = [IsAdmin]

    def get(self, request):
        labels = (
            HumanLabel.objects.select_related("pair__cv", "pair__job")
            .filter(pair__status=PairStatus.LABELED)
            .order_by("pair__split", "id")
        )

        result = []
        for hl in labels:
            result.append({
                "cv_id":          hl.pair.cv.cv_id,
                "job_id":         hl.pair.job.job_id,
                "label":          hl.binary_label,
                "split":          hl.pair.split,
                "skill_fit":      hl.skill_fit,
                "seniority_fit":  hl.seniority_fit,
                "experience_fit": hl.experience_fit,
                "domain_fit":     hl.domain_fit,
                "overall":        hl.overall,
            })

        return Response({"success": True, "count": len(result), "data": result})
