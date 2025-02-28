# csv_views.py
# import fireducks.pandas as pd
import fireducks.pandas as pd
# import polars as pl

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, JSONParser, FormParser
from rest_framework import status
from rest_framework.response import Response

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from data_processing.models import CsvModel, ProjectModel, ColumnModel
from data_processing.serializers import CsvModelSerializer, ColumnModelSerializer


class CsvView(APIView):
    '''
    CsvModel 파일 업로드, 조회, 수정 및 삭제 기능 제공
    '''
    parser_classes = [MultiPartParser, JSONParser, FormParser]

    @swagger_auto_schema(
        operation_description="csv 파일 업로드",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'csv_file': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    format='binary',  # 파일 업로드를 위한 형식
                    description="CsvModel file to upload",
                ),
                'writer': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Writer of the CsvModel file",
                ),
                'project_id': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="ID of the project to which the CsvModel file belongs",
                ),
            },
            required=['file', 'writer', 'project_id'],  # 필수 필드 정의
        ),
        responses={
            201: openapi.Response(description="File uploaded successfully"),
            400: openapi.Response(description="Invalid or empty CsvModel file"),
        },
        consumes=["multipart/form-data"],  # Swagger에서 form-data를 명시
    )
    def post(self, request, *args, **kwargs):
        csv_file = request.FILES.get("csv_file")
        writer = request.data.get("writer")
        project_id = request.data.get("project_id")

        if not csv_file or not writer or not project_id:
            return Response(
                {"error": "csv_file, writer, and project_id are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            if csv_file.name.endswith('.parquet'):
                df = pd.read_parquet(csv_file)
            else:
                df = pd.read_csv(csv_file)
        except Exception as e:
            return Response(
                {"error": f"Invalid or empty CSV file provided: {e}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if df.empty:
            return Response(
                {"error": "Empty CSV file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 프로젝트 확인
        try:
            project = ProjectModel.objects.get(id=project_id)
        except ProjectModel.DoesNotExist:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

        # CsvModel 데이터 저장
        CsvModel_serializer = CsvModelSerializer(data={
            'project': project.id,
            'csv': csv_file,
            'writer': writer,
            'size': round(csv_file.size / 1024, 2),
            'rows': len(df),
        })

        if not CsvModel_serializer.is_valid():
            return Response(CsvModel_serializer.errors, status=400)

        CsvModel_record = CsvModel_serializer.save()

        # ColumnModel 데이터 저장
        column_serializers = []
        for column in df.columns:
            column_data = {
                'csv': CsvModel_record.id,
                'column_name': column
            }
            column_serializer = ColumnModelSerializer(data=column_data)
            if not column_serializer.is_valid():
                # 트랜잭션 내에서 오류 발생 시 모든 변경 사항 롤백
                return Response(column_serializer.errors, status=400)
            column_serializers.append(column_serializer)

        # 모든 ColumnModel 저장
        for serializer in column_serializers:
            serializer.save()

        return Response(
            {'file_id': CsvModel_record.id},
            status=status.HTTP_201_CREATED
        )

    @swagger_auto_schema(
        operation_description="업로드된 csv 파일 조회",
        manual_parameters=[
            openapi.Parameter(
                "project_id",
                openapi.IN_QUERY,
                description="ID of the project to which the csv files belong",
                type=openapi.TYPE_INTEGER,
                required=True,
            ),
        ],
        responses={
            200: openapi.Response(description="CsvModel files retrieved successfully"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        프로젝트에 업로드된 csv 파일 목록 조회
        '''
        csvs = CsvModel.objects.filter(
            project=request.GET.get("project_id"))
        CsvModel_serializer = CsvModelSerializer(
            csvs, many=True)
        return Response(
            {"csvs": CsvModel_serializer.data},
            status=status.HTTP_200_OK
        )

    @swagger_auto_schema(
        operation_description="csv 파일 삭제",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'file_id': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="ID of the CsvModel file to delete"
                ),
            }
        ),
        request_body_required=True,
        responses={
            204: openapi.Response(description="File deleted successfully"),
            400: openapi.Response(description="Invalid file ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def delete(self, request, *args, **kwargs):
        '''
        csv 파일 삭제
        '''
        file_id = request.data.get("file_id")

        if not file_id or not str(file_id).isdigit():
            return Response({"error": "Invalid or missing file_id"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            CsvModel_record = CsvModel.objects.get(id=file_id)
        except CsvModel.DoesNotExist:
            return Response({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)

        CsvModel_record.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)
