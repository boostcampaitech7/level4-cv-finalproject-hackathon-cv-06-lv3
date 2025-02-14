from django.db import models
from .project_model import ProjectModel
from .csv_model import CsvModel


def surrogate_model_upload_to(instance, filename):
    return f'project_{instance.project.id}/flow_{instance.id}/surrogate_models/{filename}'


def concat_csv_upload_to(instance, filename):
    return f'project_{instance.project.id}/flow_{instance.id}/concat_csv/{filename}'


def preprocessed_csv_upload_to(instance, filename):
    return f'project_{instance.project.id}/flow_{instance.id}/preprocessed_csv/{filename}'


class FlowModel(models.Model):
    project = models.ForeignKey(
        ProjectModel, on_delete=models.CASCADE, related_name="flow_model")
    csv = models.ManyToManyField(  # Many-to-Many 관계로 변경
        CsvModel, related_name="flow_model", blank=True)
    concat_csv = models.FileField(
        upload_to=concat_csv_upload_to, blank=True, null=True)
    preprocessed_csv = models.FileField(
        upload_to=preprocessed_csv_upload_to, blank=True, null=True)
    flow_name = models.CharField(max_length=255, blank=False, null=False)
    model = models.FileField(
        upload_to=surrogate_model_upload_to, default=None, blank=True, null=True)
    progress = models.IntegerField(blank=True, null=True, default=1)

    def __str__(self):
        return self.flow_name
