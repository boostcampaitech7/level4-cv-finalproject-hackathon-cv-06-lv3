from django.db import models
from .column_model import ConcatColumnModel


class OptimizationModel(models.Model):
    # 최적화 목표
    OPTIMIZE_GOAL_CHOICES = [
        (1, 'no optimization'),  # 1. No optimization
        (2, 'Maximize'),  # 2. Maximize
        (3, 'Minimize'),  # 3. Minimize
        (4, 'Fit_to_the_range'),  # 4. Fit to the range
        (5, 'Fit_to_the_properties')  # 5. Fit to the properties
    ]
    column = models.OneToOneField(
        ConcatColumnModel, on_delete=models.CASCADE, related_name="controllable_optimizations")
    minimum_value = models.CharField(blank=False, null=False, default='0', max_length=100)
    maximum_value = models.CharField(blank=False, null=False, default='0', max_length=100)
    optimize_goal = models.IntegerField(
        choices=OPTIMIZE_GOAL_CHOICES, blank=False, null=False, default=1)
    optimize_order = models.IntegerField(blank=True, null=True, default=1)

    def __str__(self):
        return self.optimization_name
