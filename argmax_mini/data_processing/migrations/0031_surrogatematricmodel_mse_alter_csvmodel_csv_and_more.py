# Generated by Django 4.2.18 on 2025-02-05 22:25

import data_processing.models.csv_model
import data_processing.models.flow_model
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("data_processing", "0030_flowmodel_progress"),
    ]

    operations = [
        migrations.AddField(
            model_name="surrogatematricmodel",
            name="mse",
            field=models.FloatField(default=0),
        ),
        migrations.AlterField(
            model_name="csvmodel",
            name="csv",
            field=models.FileField(
                upload_to=data_processing.models.csv_model.csv_upload_to
            ),
        ),
        migrations.AlterField(
            model_name="flowmodel",
            name="concat_csv",
            field=models.FileField(
                blank=True,
                null=True,
                upload_to=data_processing.models.flow_model.concat_csv_upload_to,
            ),
        ),
        migrations.AlterField(
            model_name="flowmodel",
            name="model",
            field=models.FileField(
                blank=True,
                default=None,
                null=True,
                upload_to=data_processing.models.flow_model.surrogate_model_upload_to,
            ),
        ),
        migrations.AlterField(
            model_name="flowmodel",
            name="preprocessed_csv",
            field=models.FileField(
                blank=True,
                null=True,
                upload_to=data_processing.models.flow_model.preprocessed_csv_upload_to,
            ),
        ),
        migrations.AlterField(
            model_name="surrogatematricmodel",
            name="rmse",
            field=models.FloatField(default=0),
        ),
    ]
