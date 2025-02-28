import json

from django.core.files.base import ContentFile
from rest_framework.views import APIView
from adrf.views import APIView as ADRFAPIView
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
# import fireducks.pandas as pd
import fireducks.pandas as pd
import numpy as np

from data_processing import models
from data_processing.serializers import ConcatColumnModelSerializer, FlowModelSerializer


class FlowsView(APIView):
    '''
    Flow 조회, 생성, 삭제, 수정
    '''
    parser_classes = [JSONParser]  # 파일 업로드를 지원하는 파서 추가

    @swagger_auto_schema(
        operation_description="프로젝트의 Flow 조회",
        manual_parameters=[
            openapi.Parameter(
                "project_id",
                openapi.IN_QUERY,
                description="ID of the project",
                type=openapi.TYPE_INTEGER,
                required=True,
            ),
        ],
        responses={
            200: openapi.Response(description="Flow data retrieved successfully"),
            400: openapi.Response(description="Invalid project ID"),
            404: openapi.Response(description="Project not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        Flow 조회
        '''
        project_id = request.GET.get("project_id")

        if not project_id or not str(project_id).isdigit():
            return Response({"error": "Invalid or missing project_id"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            project = models.ProjectModel.objects.get(id=project_id)
        except models.ProjectModel.DoesNotExist:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

        flows = models.FlowModel.objects.filter(project_id=project_id)
        flows_data = FlowModelSerializer(flows, many=True).data
        # flows_data = [{"id": flow.id, "flow_name": flow.flow_name}
        #               for flow in flows]

        return Response({"flows": flows_data}, status=status.HTTP_200_OK)

    @swagger_auto_schema(
        operation_description="프로젝트의 Flow 생성",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'project_id': openapi.Schema(type=openapi.TYPE_INTEGER, description="ID of the project"),
                'flow_name': openapi.Schema(type=openapi.TYPE_STRING, description="Name of the flow"),
            }
        ),
        request_body_required=True,
        responses={
            201: openapi.Response(description="Flow created successfully"),
            400: openapi.Response(description="Invalid project ID or flow name"),
            404: openapi.Response(description="Project not found"),
        },
    )
    def post(self, request, *args, **kwargs):
        '''
        Flow 생성
        '''
        project_id = request.data.get("project_id")
        flow_name = request.data.get("flow_name")

        if not project_id or not str(project_id).isdigit():
            return Response({"error": "Invalid or missing project_id"}, status=status.HTTP_400_BAD_REQUEST)

        if not flow_name:
            return Response({"error": "No flow_name provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            project = models.ProjectModel.objects.get(id=project_id)
        except models.ProjectModel.DoesNotExist:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

        flow = models.FlowModel.objects.create(
            project=project, flow_name=flow_name)
        return Response({"flow_id": flow.id}, status=status.HTTP_201_CREATED)

    @swagger_auto_schema(
        operation_description="프로젝트의 Flow 삭제",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'flow_id': openapi.Schema(type=openapi.TYPE_INTEGER, description="ID of the flow"),
            }
        ),
        request_body_required=True,
        responses={
            200: openapi.Response(description="Flow deleted successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="Flow not found"),
        },
    )
    def delete(self, request, *args, **kwargs):
        '''
        Flow 삭제
        '''
        flow_id = request.data.get("flow_id")

        if not flow_id or not str(flow_id).isdigit():
            return Response({"error": "Invalid or missing flow_id"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            flow = models.FlowModel.objects.get(id=flow_id)
        except models.FlowModel.DoesNotExist:
            return Response({"error": "Flow not found"}, status=status.HTTP_404_NOT_FOUND)

        flow.delete()
        return Response({"flow_id": flow_id}, status=status.HTTP_200_OK)

    @swagger_auto_schema(
        operation_description="프로젝트의 Flow 수정",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'flow_id': openapi.Schema(type=openapi.TYPE_INTEGER, description="ID of the flow"),
                'flow_name': openapi.Schema(type=openapi.TYPE_STRING, description="Name of the flow"),
            }
        ),
        request_body_required=True,
        responses={
            200: openapi.Response(description="Flow updated successfully"),
            400: openapi.Response(description="Invalid flow ID or flow name"),
            404: openapi.Response(description="Flow not found"),
        },
    )
    def put(self, request, *args, **kwargs):
        '''
        Flow 이름 수정
        '''
        flow_id = request.data.get("flow_id")
        flow_name = request.data.get("flow_name")

        if not flow_id or not str(flow_id).isdigit():
            return Response({"error": "Invalid or missing flow_id"}, status=status.HTTP_400_BAD_REQUEST)

        if not flow_name:
            return Response({"error": "No flow_name provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            flow = models.FlowModel.objects.get(id=flow_id)
        except models.FlowModel.DoesNotExist:
            return Response({"error": "Flow not found"}, status=status.HTTP_404_NOT_FOUND)

        flow.flow_name = flow_name
        flow.save()
        return Response({"flow_id": flow_id, "flow_name": flow_name}, status=status.HTTP_200_OK)


class FlowCsvAddView(APIView):
    '''
    Flow에 csv 추가
    '''

    @swagger_auto_schema(
        operation_description="Flow에 csv 추가",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'flow_id': openapi.Schema(type=openapi.TYPE_INTEGER, description="ID of the flow"),
                'csv_ids': openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Schema(type=openapi.TYPE_INTEGER),
                    description="List of CSV IDs to add to the flow"
                ),
            }
        ),
        request_body_required=True,
        responses={
            200: openapi.Response(description="csv added to Flow successfully"),
            400: openapi.Response(description="Invalid flow ID or csv ID"),
            404: openapi.Response(description="Flow or csv not found"),
        },
    )
    def post(self, request, *args, **kwargs):
        '''
        Flow에 csv 추가
        '''
        flow_id = request.data.get("flow_id")
        csv_ids = request.data.get("csv_ids")

        # flow_id 유효성 검사
        if not flow_id or not isinstance(flow_id, int):
            return Response({"error": "Invalid or missing flow_id"}, status=status.HTTP_400_BAD_REQUEST)

        # csv_ids 유효성 검사
        if not csv_ids or not isinstance(csv_ids, list):
            return Response({"error": "Invalid or missing csv_ids"}, status=status.HTTP_400_BAD_REQUEST)

        # flow 존재 여부 확인
        try:
            flow = models.FlowModel.objects.get(id=flow_id)
        except models.FlowModel.DoesNotExist:
            return Response({"error": "Flow not found"}, status=status.HTTP_404_NOT_FOUND)

        # csv 존재 여부 확인 및 추가

        concat_list = []

        try:
            for csv_id in csv_ids:
                if not isinstance(csv_id, int):
                    return Response({"error": f"Invalid csv_id: {csv_id}"}, status=status.HTTP_400_BAD_REQUEST)
                csv = models.CsvModel.objects.get(id=csv_id)
                concat_list.append(csv.csv)
                flow.csv.add(csv)  # Many-to-Many 관계에 추가
        except models.CsvModel.DoesNotExist:
            return Response({"error": f"CsvModel not found for id: {csv_id}"}, status=status.HTTP_404_NOT_FOUND)

        dataframes = []
        for file in concat_list:
            if file.name.endswith('.csv'):
                dataframes.append(pd.read_csv(file, index_col=None))
            elif file.name.endswith('.parquet'):
                dataframes.append(pd.read_parquet(file))

        common_columns = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_columns = common_columns.intersection(df.columns)

        # 공통 컬럼만 선택하여 새로운 DataFrame 리스트 생성
        common_dataframes = [df[list(common_columns)] for df in dataframes]

        # 공통 컬럼을 기준으로 DataFrame 병합
        concat_df = pd.concat(common_dataframes, ignore_index=True)

        # Concatenated CSV 파일 생성
        concat_csv = ContentFile(concat_df.to_csv(index=False).encode('utf-8'))
        file_name = f"{flow.flow_name}_concat.csv"
        flow.concat_csv.save(file_name, concat_csv)

        if models.ConcatColumnModel.objects.filter(flow=flow_id).exists():
            models.ConcatColumnModel.objects.filter(flow=flow_id).delete()
            models.HistogramModel.objects.filter(
                column__flow=flow_id).delete()

        for column_name in concat_df.columns:
            series = concat_df[column_name]

            if np.issubdtype(series.dtype, np.number):
                unique_vals = series.nunique(dropna=True)
                column_type = "categorical" if unique_vals < 10 and (
                    unique_vals / len(series) < 0.005) else "numerical"
            elif series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
                avg_length = series.dropna().astype(str).apply(len).mean()
                column_type = "text" if avg_length > 50 else "categorical"
            elif any(keyword in column_name.lower() for keyword in ['date', 'time']):
                parsed = pd.to_datetime(
                    series, errors='coerce', infer_datetime_format=True)
                if parsed.notnull().sum() / len(series) >= 0.7:
                    column_type = "unavailable"
            else:
                column_type = "categorical"

            missing_values_ratio = round(
                series.isnull().mean() * 100, 2)
            if missing_values_ratio > 50:
                column_type = "unavailable"

            concat_column_serializer = ConcatColumnModelSerializer(data={
                'flow': flow.id,
                'column_name': column_name,
                'column_type': column_type,
                'property_type': 'environmental',
                'missing_values_ratio': missing_values_ratio,
            })

            if concat_column_serializer.is_valid():
                concat_column_serializer.save()
            else:
                return Response(concat_column_serializer.errors, status=400)

            # 히스토그램 데이터 저장
            if column_type == "numerical":
                concat_df_column = concat_df[column_name].dropna()
                # IQR 방식 이상치 제거
                Q1 = concat_df_column.quantile(0.25)
                Q3 = concat_df_column.quantile(0.75)
                IQR = Q3 - Q1
                concat_df_column = concat_df_column[~((concat_df_column < (Q1 - 1.5 * IQR)) | (concat_df_column > (Q3 + 1.5 * IQR)))]
                counts, bin_edges = np.histogram(concat_df_column, bins=10)

                models.HistogramModel.objects.create(
                    column=concat_column_serializer.instance,
                    counts=json.dumps(counts.tolist()),
                    bin_edges=json.dumps([round(float(i), 2) for i in bin_edges.tolist()])
                )
            elif column_type == "categorical":
                # 카테고리별 빈도 계산
                column_counts = concat_df[column_name].dropna().value_counts()
                category_counts = column_counts.tolist()
                category_names = column_counts.index.tolist()
                # 히스토그램 데이터 저장
                models.HistogramModel.objects.create(
                    column=concat_column_serializer.instance,
                    counts=json.dumps(category_counts),  # 빈도 직렬화
                    bin_edges=json.dumps(category_names)  # 카테고리 이름 직렬화
                )

        return Response(
            {"flow_id": flow_id, "csv_ids": csv_ids},
            status=status.HTTP_200_OK
        )

    @swagger_auto_schema(
        operation_description="Flow에 추가된 csv 목록 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the Flow",
            ),
        ],
        responses={
            200: openapi.Response(description="CSV files retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="Flow not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        Flow에 추가된 csv 목록 조회
        '''
        flow_id = request.GET.get("flow_id")

        if not flow_id or not str(flow_id).isdigit():
            return Response({"error": "Invalid or missing flow_id"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            flow = models.FlowModel.objects.get(id=flow_id)
        except models.FlowModel.DoesNotExist:
            return Response({"error": "Flow not found"}, status=status.HTTP_404_NOT_FOUND)

        csvs = flow.csv.all()
        csvs_data = [{"id": csv.id, "csv_name": csv.csv.name}
                     for csv in csvs]
        concat_csv = flow.concat_csv.name

        return Response({"csvs": csvs_data, 'concat_csv': concat_csv}, status=status.HTTP_200_OK)


class FlowConcatCsvView(APIView):
    '''
    Concat된 csv 파일 컬럼 데이터 조회
    '''
    @swagger_auto_schema(
        operation_description="Concat된 csv 파일 컬럼 데이터 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the Flow",
            ),
            openapi.Parameter(
                'column_name', openapi.IN_QUERY, type=openapi.TYPE_STRING,
                description="Name of the column",
            ),
        ],
        responses={
            200: openapi.Response(description="Concatenated CSV file retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="Flow not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        Concat된 csv 파일 컬럼 데이터 조회
        '''
        flow_id = request.GET.get("flow_id")
        column_name = request.GET.get("column_name")

        if not flow_id or not str(flow_id).isdigit():
            return Response({"error": "Invalid or missing flow_id"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            flow = models.FlowModel.objects.get(id=flow_id)
        except models.FlowModel.DoesNotExist:
            return Response({"error": "Flow not found"}, status=status.HTTP_404_NOT_FOUND)

        concat_column_data = pd.read_csv(flow.concat_csv)[column_name].tolist()

        return Response({"concat_column_data": concat_column_data}, status=status.HTTP_200_OK)


class FlowProgressView(ADRFAPIView):
    @swagger_auto_schema(
        operation_description="Flow 진행 상태 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the Flow",
            ),
        ],
        responses={
            200: openapi.Response(description="Flow progress retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="Flow not found"),
        },
    )
    async def get(self, request, *args, **kwargs):
        # flow_id 유효성 검사
        flow_id = request.GET.get("flow_id")
        if not flow_id or not str(flow_id).isdigit():
            return Response(
                {"error": "Invalid or missing flow_id"},
                status=status.HTTP_400_BAD_REQUEST
            )
        flow_id = int(flow_id)

        # 비동기 ORM을 사용하여 flow 조회
        try:
            flow = await models.FlowModel.objects.aget(id=flow_id)
        except models.FlowModel.DoesNotExist:
            return Response(
                {"error": "Flow not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        data = {
            "flow_id": flow.id,
            "flow_name": flow.flow_name,
            "progress": flow.progress,
        }
        return Response(data, status=status.HTTP_200_OK)
