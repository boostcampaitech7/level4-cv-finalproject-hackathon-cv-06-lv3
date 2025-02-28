from django.db import models
from .flow_model import FlowModel
from .column_model import ConcatColumnModel


class SurrogateMatricModel(models.Model):
    '''
    Output variable에 대한 surrogate model의 성능 지표
    '''
    flow = models.ForeignKey(
        FlowModel, on_delete=models.CASCADE, related_name="surrogate_matric")
    column = models.ForeignKey(
        ConcatColumnModel, on_delete=models.CASCADE, related_name="surrogate_matric")
    r_squared = models.FloatField(blank=False, null=False)
    rmse = models.FloatField(blank=False, null=True, default=0)
    mae = models.FloatField(blank=False, null=True, default=0)

    def __str__(self):
        return self.flow.flow_name


class SurrogateResultModel(models.Model):
    '''
    Output variable surrogate model의 결과
    '''
    flow = models.ForeignKey(
        FlowModel, on_delete=models.CASCADE, related_name="surrogate_result")
    column = models.ForeignKey(
        ConcatColumnModel, on_delete=models.CASCADE, related_name="surrogate_result")
    ground_truth = models.CharField(blank=False, null=False, max_length=255)
    predicted = models.CharField(blank=False, null=False, max_length=255)
    rank = models.IntegerField(blank=False, null=False)

    def __str__(self):
        return self.flow.flow_name


class FeatureImportanceModel(models.Model):
    '''
    Output variable을 예측하는데 중요한 feature importance
    '''
    flow = models.ForeignKey(
        FlowModel, on_delete=models.CASCADE, related_name="surrogate_feature_importance")
    column = models.ForeignKey(
        ConcatColumnModel, on_delete=models.CASCADE, related_name="surrogate_feature_importance")
    importance = models.FloatField(blank=False, null=False)

    def __str__(self):
        return self.flow.flow_name
