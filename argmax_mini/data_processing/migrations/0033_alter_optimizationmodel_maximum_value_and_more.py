# Generated by Django 4.2.18 on 2025-02-06 11:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data_processing', '0032_rename_mse_surrogatematricmodel_mae'),
    ]

    operations = [
        migrations.AlterField(
            model_name='optimizationmodel',
            name='maximum_value',
            field=models.CharField(default='0', max_length=100),
        ),
        migrations.AlterField(
            model_name='optimizationmodel',
            name='minimum_value',
            field=models.CharField(default='0', max_length=100),
        ),
    ]
