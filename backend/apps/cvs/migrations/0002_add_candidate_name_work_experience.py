from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cvs", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="cv",
            name="candidate_name",
            field=models.CharField(blank=True, default="", max_length=200),
        ),
        migrations.AddField(
            model_name="cv",
            name="work_experience",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
