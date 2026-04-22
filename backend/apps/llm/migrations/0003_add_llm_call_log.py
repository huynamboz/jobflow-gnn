import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("llm", "0002_add_client_type"),
    ]

    operations = [
        migrations.CreateModel(
            name="LLMCallLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("provider", models.ForeignKey(
                    blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL,
                    related_name="call_logs", to="llm.llmprovider",
                )),
                ("feature", models.CharField(blank=True, default="", max_length=100)),
                ("status", models.CharField(choices=[("success", "Success"), ("error", "Error")], max_length=20)),
                ("input_preview", models.CharField(blank=True, default="", max_length=1000)),
                ("output", models.TextField(blank=True, default="")),
                ("error_message", models.TextField(blank=True, default="")),
                ("duration_ms", models.IntegerField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"db_table": "llm_call_logs", "ordering": ["-created_at"]},
        ),
    ]
