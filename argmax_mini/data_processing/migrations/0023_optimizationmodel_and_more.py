# Generated by Django 4.2.18 on 2025-02-02 00:28

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data_processing', '0022_rename_target_value_outputoptimizationmodel_maximum_value_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='OptimizationModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('minimum_value', models.FloatField(default=0)),
                ('maximum_value', models.FloatField(default=0)),
                ('optimize_goal', models.IntegerField(choices=[(1, 'no optimization'), (2, 'Maximize'), (3, 'Minimize'), (4, 'Fit_to_the_range'), (5, 'Fit_to_the_properties')], default=1)),
                ('column', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='controllable_optimizations', to='data_processing.concatcolumnmodel')),
            ],
        ),
        migrations.RemoveField(
            model_name='outputoptimizationmodel',
            name='column',
        ),
        migrations.DeleteModel(
            name='ControllableOptimizationModel',
        ),
        migrations.DeleteModel(
            name='OutputOptimizationModel',
        ),
    ]
