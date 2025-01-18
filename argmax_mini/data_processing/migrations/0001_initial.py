# Generated by Django 4.2.18 on 2025-01-17 03:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ColumnRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('column_name', models.CharField(max_length=255)),
                ('is_unique', models.BooleanField(default=False)),
                ('column_type', models.CharField(choices=[('numerical', 'Numerical'), ('categorical', 'Categorical'), ('unavailable', 'Unavailable')], default='unavailable', max_length=13)),
                ('property_type', models.CharField(choices=[('environmental', 'Environmental'), ('controllable', 'Controllable'), ('output', 'Output')], default='environmental', max_length=13)),
            ],
        ),
        migrations.CreateModel(
            name='CsvDataRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_name', models.CharField(max_length=255)),
                ('data', models.JSONField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='HistogramRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('counts', models.JSONField()),
                ('bin_edges', models.JSONField()),
                ('column_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='histogram_record', to='data_processing.columnrecord')),
            ],
        ),
        migrations.AddField(
            model_name='columnrecord',
            name='csv_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='column_record', to='data_processing.csvdatarecord'),
        ),
    ]
