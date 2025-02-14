from django.contrib import admin

from data_processing.models import ConcatColumnModel, CsvModel, HistogramModel, ProjectModel, FlowModel, SearchResultModel, SurrogateMatricModel, SurrogateResultModel, FeatureImportanceModel, OptimizationModel


class ProjectModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'created_at')


class CsvModelAdmin(admin.ModelAdmin):
    list_display = ('project', 'csv', 'writer', 'size', 'rows', 'created_at')


class FlowModelAdmin(admin.ModelAdmin):
    list_display = ('project', 'get_csv', 'concat_csv', 'flow_name')

    def get_csv(self, obj):
        return ", ".join([csv.csv.name for csv in obj.csv.all()])


class HistogramModelAdmin(admin.ModelAdmin):
    list_display = ('column', 'counts', 'bin_edges')


class ConcatColumnModelAdmin(admin.ModelAdmin):
    list_display = ('flow', 'column_name', 'column_type',
                    'property_type', 'missing_values_ratio')


class SearchResultModelAdmin(admin.ModelAdmin):
    list_display = ('flow', 'column', 'ground_truth', 'predicted')


class SurrogateMatricModelAdmin(admin.ModelAdmin):
    list_display = ('flow', 'column', 'r_squared', 'rmse', 'mae')


class SurrogateResultModelAdmin(admin.ModelAdmin):
    list_display = ('flow', 'column', 'ground_truth', 'predicted', 'rank')


class FeatureImportanceModelAdmin(admin.ModelAdmin):
    list_display = ('flow', 'column', 'importance')


class OptimizationModelAdmin(admin.ModelAdmin):
    list_display = ('column', 'minimum_value',
                    'maximum_value', 'optimize_goal')


admin.site.register(ProjectModel, ProjectModelAdmin)
admin.site.register(CsvModel, CsvModelAdmin)
admin.site.register(HistogramModel, HistogramModelAdmin)
admin.site.register(ConcatColumnModel, ConcatColumnModelAdmin)
admin.site.register(FlowModel, FlowModelAdmin)
admin.site.register(SearchResultModel, SearchResultModelAdmin)
admin.site.register(SurrogateMatricModel, SurrogateMatricModelAdmin)
admin.site.register(SurrogateResultModel, SurrogateResultModelAdmin)
admin.site.register(FeatureImportanceModel, FeatureImportanceModelAdmin)
admin.site.register(OptimizationModel,
                    OptimizationModelAdmin)
