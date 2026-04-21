from rest_framework import serializers

from .models import HumanLabel, LabelingCV, LabelingJob, PairQueue


class LabelingCVSerializer(serializers.ModelSerializer):
    class Meta:
        model = LabelingCV
        fields = ["cv_id", "source", "skills", "seniority", "experience_years", "education", "text_summary", "pdf_path"]


class LabelingJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = LabelingJob
        fields = ["job_id", "title", "skills", "seniority", "salary_min", "salary_max", "text_summary"]


class PairQueueItemSerializer(serializers.ModelSerializer):
    job = LabelingJobSerializer(read_only=True)
    common_skills = serializers.SerializerMethodField()

    class Meta:
        model = PairQueue
        fields = ["id", "job", "skill_overlap_score", "selection_reason", "split", "common_skills"]

    def get_common_skills(self, obj):
        cv_skills = set(obj.cv.skills)
        job_skills = set(obj.job.skills)
        return sorted(cv_skills & job_skills)


class QueueResponseSerializer(serializers.Serializer):
    cv = LabelingCVSerializer()
    pairs = PairQueueItemSerializer(many=True)
    progress = serializers.DictField()


class SubmitLabelSerializer(serializers.Serializer):
    skill_fit      = serializers.ChoiceField(choices=[0, 1, 2])
    seniority_fit  = serializers.ChoiceField(choices=[0, 1, 2])
    experience_fit = serializers.ChoiceField(choices=[0, 1, 2])
    domain_fit     = serializers.ChoiceField(choices=[0, 1, 2])
    overall        = serializers.ChoiceField(choices=[0, 1, 2])
    note           = serializers.CharField(allow_blank=True, required=False, default="")


class HumanLabelSerializer(serializers.ModelSerializer):
    binary_label = serializers.ReadOnlyField()

    class Meta:
        model = HumanLabel
        fields = [
            "id", "pair_id", "skill_fit", "seniority_fit", "experience_fit",
            "domain_fit", "overall", "binary_label", "note", "created_at",
        ]


class ExportItemSerializer(serializers.Serializer):
    cv_id          = serializers.IntegerField()
    job_id         = serializers.IntegerField()
    label          = serializers.IntegerField()
    split          = serializers.CharField()
    skill_fit      = serializers.IntegerField()
    seniority_fit  = serializers.IntegerField()
    experience_fit = serializers.IntegerField()
    domain_fit     = serializers.IntegerField()
    overall        = serializers.IntegerField()
