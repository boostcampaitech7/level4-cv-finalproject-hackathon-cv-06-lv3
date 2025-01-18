# Generated by Django 4.2.18 on 2025-01-17 14:41

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data_processing', '0002_project_alter_columnrecord_column_name_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='csvdatarecord',
            name='project_id',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='csv_data_record', to='data_processing.project'),
            preserve_default=False,
        ),
    ]
