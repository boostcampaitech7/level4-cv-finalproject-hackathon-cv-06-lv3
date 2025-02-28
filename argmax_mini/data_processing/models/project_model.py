from django.db import models


class ProjectModel(models.Model):
    name = models.CharField(max_length=20, blank=False, null=False)
    description = models.CharField(max_length=50, blank=False, null=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
