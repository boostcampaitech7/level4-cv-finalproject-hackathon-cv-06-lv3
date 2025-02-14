from rest_framework import serializers
from data_processing import models


class ProjectModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ProjectModel
        fields = '__all__'


class CsvModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.CsvModel
        fields = '__all__'


class ColumnModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ColumnModel
        fields = '__all__'


class ConcatColumnModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ConcatColumnModel
        fields = '__all__'


class HistogramModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.HistogramModel
        fields = '__all__'


class FlowModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.FlowModel
        fields = '__all__'


class OptimizationModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.OptimizationModel
        fields = '__all__'


class SurrogateResultModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SurrogateResultModel
        fields = '__all__'


class SurrogateMatricModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SurrogateMatricModel
        fields = '__all__'


class FeatureImportanceModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.FeatureImportanceModel
        fields = '__all__'


class SearchResultModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.SearchResultModel
        fields = '__all__'
