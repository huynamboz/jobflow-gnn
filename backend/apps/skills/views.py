"""Skill endpoints (/api/skills/)."""

from drf_spectacular.utils import extend_schema
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.skills.models import Skill
from apps.skills.serializers import SkillSerializer


class SkillListView(APIView):
    """GET /api/skills/ — List all skills (public)."""

    permission_classes = [AllowAny]

    @extend_schema(responses={200: SkillSerializer(many=True)}, tags=["Skills"])
    def get(self, request):
        category = request.query_params.get("category")
        search = request.query_params.get("search")

        qs = Skill.objects.all()
        if category is not None and category != "":
            qs = qs.filter(category=category)
        if search:
            qs = qs.filter(canonical_name__icontains=search)

        return Response({"success": True, "data": SkillSerializer(qs, many=True).data})
