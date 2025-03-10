# Generated by Django 4.2.18 on 2025-01-28 00:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data_processing', '0009_remove_concatcolumnmodel_optimize_goal_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='OptimizationModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('minimum_value', models.FloatField(default=0)),
                ('maximum_value', models.FloatField(default=0)),
                ('optimize_goal', models.IntegerField(choices=[(1, 'Do_not_optimize'), (2, 'Maximize'), (3, 'Minimize'), (4, 'Fit_to_the_range'), (5, 'Fit_to_the_properties')], default=1)),
                ('column', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='optimizations', to='data_processing.concatcolumnmodel')),
            ],
        ),
    ]
