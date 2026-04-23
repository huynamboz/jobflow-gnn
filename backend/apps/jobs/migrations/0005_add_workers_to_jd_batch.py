from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("jobs", "0004_add_source_url_to_jd_record"),
    ]

    operations = [
        migrations.AddField(
            model_name="jdextractionbatch",
            name="workers",
            field=models.PositiveSmallIntegerField(default=3),
        ),
    ]
