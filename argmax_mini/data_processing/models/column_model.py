from django.db import models
from .flow_model import CsvModel, FlowModel


class ColumnModel(models.Model):
    csv = models.ForeignKey(
        CsvModel, on_delete=models.CASCADE, related_name="columns")
    column_name = models.CharField(max_length=50, blank=False, null=False)


class ConcatColumnModel(models.Model):
    COLUMN_TYPE_CHOICES = [
        ("numerical", "Numerical"),
        ("categorical", "Categorical"),
        ('text', 'Text'),
        ('unavailable', 'Unavailable')
    ]

    PROPERTY_TYPE_CHOICES = [
        ('environmental', 'Environmental'),
        ('controllable', 'Controllable'),
        ('output', 'Output'),
    ]

    flow = models.ForeignKey(
        FlowModel, on_delete=models.CASCADE, related_name="columns")
    column_name = models.CharField(max_length=50, blank=False, null=False)
    column_type = models.CharField(
        max_length=13, choices=COLUMN_TYPE_CHOICES, blank=False, null=False,
        default='unavailable')
    property_type = models.CharField(
        max_length=13, choices=PROPERTY_TYPE_CHOICES, blank=False,
        null=False, default='environmental')
    missing_values_ratio = models.FloatField(
        blank=False, null=False, default=0)

    def __str__(self):
        return self.column_name
