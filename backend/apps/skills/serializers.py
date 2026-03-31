from rest_framework import serializers

from apps.skills.models import Skill


class SkillSerializer(serializers.ModelSerializer):
    class Meta:
        model = Skill
        fields = ("id", "canonical_name", "category")
