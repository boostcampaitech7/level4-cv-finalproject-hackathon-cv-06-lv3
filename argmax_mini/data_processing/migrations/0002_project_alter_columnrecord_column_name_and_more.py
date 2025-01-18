# Generated by Django 4.2.18 on 2025-01-17 14:35

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('data_processing', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Project',
            fields=[('id', models.BigAutoField(
                auto_created=True, primary_key=True, serialize=False,
                verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
                ('description', models.CharField(max_length=50)),
                ('created_at', models.DateTimeField(
                    auto_now_add=True)),],),
        migrations.AlterField(
            model_name='columnrecord', name='column_name',
            field=models.CharField(max_length=50),),]
