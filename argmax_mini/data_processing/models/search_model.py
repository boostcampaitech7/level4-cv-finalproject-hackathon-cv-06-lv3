from django.db import models
from .flow_model import FlowModel
from .column_model import ConcatColumnModel


class SearchResultModel(models.Model):
    # output의 ground truth는 search model에 넣는 Y 값이고
    # output의 predicted는 search model에서 Y값을 이용해 찾은 X 값으로, 다시 surrogate model을 통해 예측한 Y 값이다.
    flow = models.ForeignKey(
        FlowModel, on_delete=models.CASCADE, related_name="search_result")
    column = models.ForeignKey(
        ConcatColumnModel, on_delete=models.CASCADE, related_name="search_result")
    ground_truth = models.JSONField()
    predicted = models.JSONField()
    average_change_rate = models.FloatField(default=0.0)

    def __str__(self):
        return self.search_result
