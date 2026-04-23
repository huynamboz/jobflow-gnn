from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("labeling", "0002_labelingcv_pdf_path"),
    ]

    operations = [
        migrations.AddField(
            model_name="labelingcv",
            name="role_category",
            field=models.CharField(blank=True, default="other", max_length=20),
        ),
        migrations.AddField(
            model_name="labelingjob",
            name="role_category",
            field=models.CharField(blank=True, default="other", max_length=20),
        ),
        migrations.AddField(
            model_name="labelingjob",
            name="experience_min",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="labelingjob",
            name="experience_max",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.CreateModel(
            name="LabelingBatch",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("status", models.CharField(
                    choices=[("running", "Running"), ("done", "Done"), ("error", "Error"), ("cancelled", "Cancelled")],
                    default="running",
                    max_length=20,
                )),
                ("total",       models.IntegerField(default=0)),
                ("done_count",  models.IntegerField(default=0)),
                ("error_count", models.IntegerField(default=0)),
                ("workers",     models.PositiveSmallIntegerField(default=3)),
                ("created_at",  models.DateTimeField(auto_now_add=True)),
            ],
            options={"db_table": "labeling_batches", "ordering": ["-created_at"]},
        ),
    ]
