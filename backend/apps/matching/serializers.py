from rest_framework import serializers


class CVTextMatchRequest(serializers.Serializer):
    text = serializers.CharField(min_length=10, help_text="CV text content")
    top_k = serializers.IntegerField(default=10, min_value=1, max_value=100)


class CVFileMatchRequest(serializers.Serializer):
    file = serializers.FileField(help_text="CV file (PDF/DOCX/TXT)")
    top_k = serializers.IntegerField(default=10, min_value=1, max_value=100, required=False)

    class Meta:
        # Tell drf-spectacular this is multipart
        pass


class JobMatchResponse(serializers.Serializer):
    job_id = serializers.IntegerField()
    score = serializers.FloatField()
    eligible = serializers.BooleanField()
    matched_skills = serializers.ListField(child=serializers.CharField())
    missing_skills = serializers.ListField(child=serializers.CharField())
    seniority_match = serializers.BooleanField()
    title = serializers.CharField()


class CVParseResponse(serializers.Serializer):
    seniority = serializers.CharField()
    experience_years = serializers.FloatField()
    education = serializers.CharField()
    skills = serializers.ListField(child=serializers.CharField())
