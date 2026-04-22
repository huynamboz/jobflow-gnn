from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("llm", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmprovider",
            name="client_type",
            field=models.CharField(
                choices=[
                    ("openai", "OpenAI Compatible (/chat/completions)"),
                    ("messages", "Messages API (/messages)"),
                ],
                default="openai",
                max_length=20,
            ),
        ),
    ]
