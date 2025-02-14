from django.db import models
from .column_model import ConcatColumnModel


class HistogramModel(models.Model):
    column = models.OneToOneField(
        ConcatColumnModel, on_delete=models.CASCADE,
        related_name="histograms")
    counts = models.JSONField()
    bin_edges = models.JSONField()
