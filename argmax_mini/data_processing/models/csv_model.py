from django.db import models
from .project_model import ProjectModel


def csv_upload_to(instance, filename):
    return f'project_{instance.project.id}/csv_files/{filename}'


class CsvModel(models.Model):
    project = models.ForeignKey(
        ProjectModel, on_delete=models.CASCADE, related_name="CsvModel_files")
    csv = models.FileField(
        upload_to=csv_upload_to, blank=False, null=False)
    writer = models.CharField(max_length=20, blank=False, null=False)
    size = models.FloatField(blank=False, null=False, default=0)
    rows = models.IntegerField(blank=False, null=False, default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.csv.name
