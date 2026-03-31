from rest_framework import serializers

from apps.cvs.models import CV, CVSkill


class CVSkillSerializer(serializers.ModelSerializer):
    skill_name = serializers.CharField(source="skill.canonical_name", read_only=True)
    category = serializers.IntegerField(source="skill.category", read_only=True)

    class Meta:
        model = CVSkill
        fields = ("skill_name", "category", "proficiency")


class CVListSerializer(serializers.ModelSerializer):
    skill_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = CV
        fields = ("id", "file_name", "seniority", "experience_years", "education", "source", "skill_count", "is_active", "created_at")


class CVDetailSerializer(serializers.ModelSerializer):
    skills = CVSkillSerializer(source="cv_skills", many=True, read_only=True)

    class Meta:
        model = CV
        fields = (
            "id", "file_name", "seniority", "experience_years", "education",
            "parsed_text", "source", "source_category", "skills",
            "is_active", "created_at", "updated_at",
        )
