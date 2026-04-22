from rest_framework import serializers

from apps.llm.models import LLMCallLog, LLMProvider


class LLMProviderSerializer(serializers.ModelSerializer):
    """Read serializer — API key is masked."""

    api_key = serializers.SerializerMethodField()

    class Meta:
        model = LLMProvider
        fields = ["id", "name", "api_key", "model", "base_url", "client_type", "is_active", "created_at", "updated_at"]
        read_only_fields = ["id", "is_active", "created_at", "updated_at"]

    def get_api_key(self, obj: LLMProvider) -> str:
        return obj.masked_api_key()


class LLMProviderWriteSerializer(serializers.ModelSerializer):
    """Write serializer — accepts full API key, validates uniqueness."""

    class Meta:
        model = LLMProvider
        fields = ["name", "api_key", "model", "base_url", "client_type"]

    def validate_name(self, value: str) -> str:
        qs = LLMProvider.objects.filter(name=value)
        if self.instance:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError("A provider with this name already exists.")
        return value

    def validate_api_key(self, value: str) -> str:
        if not value.strip():
            raise serializers.ValidationError("API key cannot be empty.")
        return value.strip()

    def validate_base_url(self, value: str) -> str:
        if not value.startswith("http"):
            raise serializers.ValidationError("base_url must start with http:// or https://")
        return value.rstrip("/")


class LLMProviderTestSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    message = serializers.CharField()


class LLMCallLogSerializer(serializers.ModelSerializer):
    provider_name = serializers.CharField(source="provider.name", read_only=True, default=None)

    class Meta:
        model = LLMCallLog
        fields = [
            "id", "provider_name", "feature", "status",
            "input_preview", "output", "error_message",
            "duration_ms", "created_at",
        ]
