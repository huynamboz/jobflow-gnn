import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("labeling", "0003_add_role_category_and_batch"),
    ]

    operations = [
        migrations.AddField(
            model_name="humanlabel",
            name="batch",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="labels",
                to="labeling.labelingbatch",
            ),
        ),
    ]
