from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("labeling", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="labelingcv",
            name="pdf_path",
            field=models.CharField(blank=True, default="", max_length=500),
        ),
    ]
