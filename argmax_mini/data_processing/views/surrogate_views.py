from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from data_processing.models import SurrogateMatricModel, SurrogateResultModel, FeatureImportanceModel
from data_processing.serializers import SurrogateMatricModelSerializer, SurrogateResultModelSerializer, FeatureImportanceModelSerializer

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class SurrogateMatricView(APIView):
    '''
    Surrogate Model Metric 조회
    '''
    @swagger_auto_schema(
        operation_description="Surrogate Model Metric 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the flow",
            ),
        ],
        responses={
            200: openapi.Response(description="Surrogate Model Metric retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        Surrogate Model Metric 조회
        '''

        flow_id = request.GET.get("flow_id")

        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not SurrogateMatricModel.objects.filter(flow=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        surrogate_matric = SurrogateMatricModel.objects.filter(flow_id=flow_id)
        serialized_data = SurrogateMatricModelSerializer(
            surrogate_matric, many=True).data
        for item, instance in zip(serialized_data, surrogate_matric):
            item['column_name'] = instance.column.column_name
            item['column_type'] = instance.column.column_type
            item['property_type'] = instance.column.property_type

        return Response({"surrogate_matric": serialized_data}, status=200)


class SurrogateResultView(APIView):
    '''
    Surrogate Model Result 조회
    '''
    @swagger_auto_schema(
        operation_description="Surrogate Model Result 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the flow",
            ),
        ],
        responses={
            200: openapi.Response(description="Surrogate Model Result retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        Surrogate Model Result 조회
        '''

        flow_id = request.GET.get("flow_id")

        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not SurrogateResultModel.objects.filter(flow=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        surrogate_result = SurrogateResultModel.objects.filter(flow=flow_id)
        serialized_data = SurrogateResultModelSerializer(
            surrogate_result, many=True).data
        
        # serialized_data에서 ground_truth와 predicted가 숫자인 경우 float로 변환
        for item in serialized_data:
            if is_number(item['ground_truth']):
                item['ground_truth'] = float(item['ground_truth'])
            if is_number(item['predicted']):
                item['predicted'] = float(item['predicted'])

        for item, instance in zip(serialized_data, surrogate_result):
            item['column_name'] = instance.column.column_name
            item['column_type'] = instance.column.column_type
            item['property_type'] = instance.column.property_type

        return Response({"surrogate_result": serialized_data}, status=200)


class FeatureImportanceView(APIView):
    '''
    Feature Importance 조회
    '''
    @swagger_auto_schema(
        operation_description="Feature Importance 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the flow",
            ),
        ],
        responses={
            200: openapi.Response(description="Feature Importance retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        Feature Importance 조회
        '''

        flow_id = request.GET.get("flow_id")

        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not FeatureImportanceModel.objects.filter(flow=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        surrogate_feature_importance = FeatureImportanceModel.objects.filter(
            flow=flow_id)

        serialized_data = FeatureImportanceModelSerializer(
            surrogate_feature_importance, many=True).data

        # Add 'column_name' for each item by iterating over both the serialized data and the queryset
        for item, instance in zip(serialized_data, surrogate_feature_importance):
            item['column_name'] = instance.column.column_name
            item['column_type'] = instance.column.column_type
            item['property_type'] = instance.column.property_type

        return Response({"feature_importance": serialized_data}, status=200)
