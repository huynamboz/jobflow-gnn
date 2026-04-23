from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("jobs", "0005_add_workers_to_jd_batch"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="role_category",
            field=models.CharField(max_length=20, blank=True, default="other"),
        ),
        migrations.AddField(
            model_name="job",
            name="experience_min",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="job",
            name="experience_max",
            field=models.FloatField(null=True, blank=True),
        ),
    ]
