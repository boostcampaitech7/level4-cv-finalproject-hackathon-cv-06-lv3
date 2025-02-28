# import fireducks.pandas as pd
import fireducks.pandas as pd

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from data_processing.models import ConcatColumnModel, FlowModel
from data_processing.serializers import HistogramModelSerializer


class HistogramView(APIView):
    '''
    Concat된 csv 파일의 히스토그램 데이터 조회
    '''
    @swagger_auto_schema(
        operation_description="concat된 csv의 각 컬럼별 히스토그램 데이터 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the Concated csv file",
            ),
            openapi.Parameter(
                'column_name', openapi.IN_QUERY, type=openapi.TYPE_STRING,
                description="Name of the column",
            ),
        ],
        responses={
            200: openapi.Response(description="Histogram data retrieved successfully"),
            400: openapi.Response(description="Invalid file ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        특정 컬럼별 히스토그램 데이터 조회
        '''

        flow_id = request.GET.get("flow_id")
        column_name = request.GET.get("column_name")

        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not FlowModel.objects.filter(id=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        if not column_name:
            print(1)
            return Response({"error": "File not found"}, status=404)

        # 데이터베이스에서 해당 파일의 히스토그램 데이터 가져오기
        try:
            columns = ConcatColumnModel.objects.filter(
                flow=flow_id, column_name=column_name).first()
        except ConcatColumnModel.DoesNotExist:
            return Response({"error": "File not found"}, status=404)

        histogram_serializer = HistogramModelSerializer(columns.histograms)

        return Response(
            {"histograms": histogram_serializer.data},
            status=200
        )


class HistogramAllView(APIView):
    @swagger_auto_schema(
        operation_description="concat된 csv의 모든 컬럼 히스토그램 데이터 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the Concated csv file",
            ),
        ],
        responses={
            200: openapi.Response(description="Histogram data retrieved successfully"),
            400: openapi.Response(description="Invalid file ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        모든 컬럼 히스토그램 데이터 조회
        '''

        flow_id = request.GET.get("flow_id")
        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not FlowModel.objects.filter(id=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        try:
            columns = ConcatColumnModel.objects.filter(flow=flow_id)
        except ConcatColumnModel.DoesNotExist:
            return Response({"error": "File not found"}, status=404)

        context = {}
        for column in columns:
            try:
                histogram_serializer = HistogramModelSerializer(
                    column.histograms)
                context[column.column_name] = histogram_serializer.data
            except:
                context[column.column_name] = []

        return Response(
            {"histograms": context},
            status=200
        )


class DataCleaningView(APIView):
    '''
    Data Cleaning Ratio 조회
    '''

    @swagger_auto_schema(
        operation_description="Data Cleaning Ratio 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the flow",
            ),
        ],
        responses={
            200: openapi.Response(description="Data Cleaning Ratio retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        Data Cleaning Ratio 조회
        '''

        flow_id = request.GET.get("flow_id")

        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not FlowModel.objects.filter(id=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        flow = FlowModel.objects.get(id=flow_id)

        origin_rows = pd.read_csv(flow.concat_csv).shape[0]

        # row 개수
        after_cleaning_rows = pd.read_csv(flow.preprocessed_csv).shape[0]

        data_cleaning_ratio = (origin_rows - after_cleaning_rows) / origin_rows
        using_data_ratio = 1 - data_cleaning_ratio

        return Response({"data_cleaning_ratio": data_cleaning_ratio, "using_data_ratio": using_data_ratio}, status=200)
