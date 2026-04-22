from django.db import models


class LLMProvider(models.Model):
    """An OpenAI-compatible LLM provider (OpenAI, Anthropic, Gemini, Ollama, …)."""

    CLIENT_OPENAI = "openai"
    CLIENT_MESSAGES = "messages"
    CLIENT_CHOICES = [
        (CLIENT_OPENAI, "OpenAI Compatible (/chat/completions)"),
        (CLIENT_MESSAGES, "Messages API (/messages)"),
    ]

    name = models.CharField(max_length=100, unique=True)
    api_key = models.CharField(max_length=500)
    model = models.CharField(max_length=100, help_text="e.g. gpt-4o-mini, claude-haiku-4-5")
    base_url = models.URLField(max_length=500, help_text="Base URL, e.g. https://api.openai.com/v1")
    client_type = models.CharField(max_length=20, choices=CLIENT_CHOICES, default=CLIENT_OPENAI)
    is_active = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "llm_providers"
        ordering = ["-is_active", "name"]

    def __str__(self) -> str:
        active = " [active]" if self.is_active else ""
        return f"{self.name} ({self.model}){active}"

    def masked_api_key(self) -> str:
        """Return API key with all but last 4 chars masked."""
        if len(self.api_key) <= 4:
            return "****"
        return f"{'*' * (len(self.api_key) - 4)}{self.api_key[-4:]}"


class LLMCallLog(models.Model):
    STATUS_SUCCESS = "success"
    STATUS_ERROR = "error"
    STATUS_CHOICES = [
        (STATUS_SUCCESS, "Success"),
        (STATUS_ERROR, "Error"),
    ]

    provider = models.ForeignKey(
        LLMProvider, null=True, blank=True, on_delete=models.SET_NULL, related_name="call_logs",
    )
    feature = models.CharField(max_length=100, blank=True, default="")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    input_preview = models.CharField(max_length=1000, blank=True, default="")
    output = models.TextField(blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    duration_ms = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "llm_call_logs"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"[{self.status}] {self.feature or 'unknown'} ({self.duration_ms}ms)"
