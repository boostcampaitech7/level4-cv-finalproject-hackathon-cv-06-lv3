# Generated by Django 4.2.18 on 2025-01-31 19:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data_processing', '0017_surrogatefeatureimportancemodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='FeatureImportanceModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('importance', models.FloatField()),
                ('column', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='surrogate_feature_importance', to='data_processing.concatcolumnmodel')),
                ('flow', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='surrogate_feature_importance', to='data_processing.flowmodel')),
            ],
        ),
        migrations.AlterField(
            model_name='surrogatematricmodel',
            name='flow',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='surrogate_matric', to='data_processing.flowmodel'),
        ),
        migrations.DeleteModel(
            name='SurrogateFeatureImportanceModel',
        ),
    ]
