from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='JDExtractionBatch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_path', models.CharField(max_length=1000)),
                ('fields_config', models.JSONField(default=list)),
                ('status', models.CharField(
                    choices=[('pending', 'Pending'), ('running', 'Running'), ('done', 'Done'), ('error', 'Error'), ('cancelled', 'Cancelled')],
                    default='pending', max_length=20,
                )),
                ('total', models.IntegerField(default=0)),
                ('done_count', models.IntegerField(default=0)),
                ('error_count', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'db_table': 'jd_extraction_batches', 'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='JDExtractionRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('batch', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='records', to='jobs.jdextractionbatch')),
                ('index', models.IntegerField()),
                ('raw_data', models.JSONField(default=dict)),
                ('combined_text', models.TextField(blank=True, default='')),
                ('status', models.CharField(
                    choices=[('pending', 'Pending'), ('processing', 'Processing'), ('done', 'Done'), ('error', 'Error')],
                    default='pending', max_length=20,
                )),
                ('result', models.JSONField(blank=True, null=True)),
                ('error_msg', models.CharField(blank=True, default='', max_length=500)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'db_table': 'jd_extraction_records', 'ordering': ['index']},
        ),
    ]
