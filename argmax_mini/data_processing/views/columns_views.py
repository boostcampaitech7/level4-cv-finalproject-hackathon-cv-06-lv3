from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from data_processing.models import ColumnModel, ConcatColumnModel


class ColumnView(APIView):
    @swagger_auto_schema(
        operation_description="csv 파일의 컬럼명 조회",
        manual_parameters=[
            openapi.Parameter(
                "csv_id",
                openapi.IN_QUERY,
                description="ID of the csv file",
                type=openapi.TYPE_INTEGER,
                required=True,
            ),
        ],
        responses={
            200: openapi.Response(description="Column names retrieved successfully"),
            400: openapi.Response(description="Invalid csv ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        csv 파일의 컬럼명 조회
        '''
        csv_id = request.GET.get("csv_id")

        if not csv_id or not csv_id.isdigit():
            return Response({"error": "No csv_id provided"}, status=400)

        if not ColumnModel.objects.filter(csv=csv_id).exists():
            return Response({"error": "File not found"}, status=404)

        column_names = ColumnModel.objects.filter(
            csv=csv_id).values_list('column_name', flat=True)

        return Response({"column_names": list(column_names)}, status=200)


class ConcatColumnPropertiesView(APIView):
    '''
    concat된 csv 파일의 컬럼 속성 조회 및 수정
    '''
    parser_classes = [JSONParser]  # 파일 업로드를 지원하는 파서 추가

    @swagger_auto_schema(
        operation_description="concat된 csv 파일의 컬럼 속성 조회",
        manual_parameters=[
            openapi.Parameter(
                "flow_id",
                openapi.IN_QUERY,
                description="Flow ID",
                type=openapi.TYPE_INTEGER,
                required=True,
            ),
        ],
        responses={
            200: openapi.Response(description="Column columns retrieved successfully"),
            400: openapi.Response(description="Invalid file ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        컬럼 속성 조회
        '''
        column_type = ConcatColumnModel.COLUMN_TYPE_CHOICES
        property_type = ConcatColumnModel.PROPERTY_TYPE_CHOICES
        flow_id = request.GET.get("flow_id")

        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not ConcatColumnModel.objects.filter(flow=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        context = {}

        for i in column_type:
            # QuerySet을 list로 변환
            context[i[0]] = list(ConcatColumnModel.objects.filter(
                flow=flow_id,
                column_type=i[0]).values_list('column_name', flat=True))

        for i in property_type:
            # QuerySet을 list로 변환
            context[i[0]] = list(ConcatColumnModel.objects.filter(
                flow=flow_id,
                property_type=i[0]).exclude(column_type="unavailable").values_list('column_name', flat=True))

        return Response(context, status=200)

    @swagger_auto_schema(
        operation_description="concat된 csv파일의 property_type 속성 수정",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'flow_id': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Flow ID"
                ),
                'column_name': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="수정하려는 컬럼의 이름"
                ),
                'property_type': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    enum=["environmental", "controllable",
                          "output"],  # 선택 가능한 옵션
                    description="컬럼의 새로운 타입 (선택 가능)"
                ),
            }
        ),
        request_body_required=True,
        responses={
            200: openapi.Response(description="Column properties updated successfully"),
            400: openapi.Response(description="Invalid request body"),
            404: openapi.Response(description="Column or File not found"),
        },
    )
    def put(self, request, *args, **kwargs):
        '''
        property_type 속성 수정
        '''
        column_name = str(request.data.get("column_name"))
        property_type = str(request.data.get("property_type"))
        flow_id = str(request.data.get("flow_id"))

        if column_name == 'None' or property_type == 'None' or flow_id == 'None':
            return Response(
                {"error": "Both column_name and property_type are required"},
                status=400)

        if not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not ConcatColumnModel.objects.filter(flow=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        column = ConcatColumnModel.objects.filter(
            column_name=column_name, flow=flow_id).first()

        if not column:
            return Response({"error": "Column not found"}, status=404)

        column.property_type = property_type
        column.save()

        return Response(
            {"column_name": column.column_name,
             "property_type": column.property_type},
            status=200)

class ConcatColumnTypeView(APIView):
    '''
    concat된 csv 파일의 컬럼 타입 조회 및 수정
    '''
    parser_classes = [JSONParser]

    @swagger_auto_schema(
        operation_description="concat된 csv 파일의 컬럼 타입 조회",
        manual_parameters=[
            openapi.Parameter(
                "flow_id",
                openapi.IN_QUERY,
                description="Flow ID",
                type=openapi.TYPE_INTEGER,
                required=True,
            ),
        ],
        responses={
            200: openapi.Response(description="Column columns retrieved successfully"),
            400: openapi.Response(description="Invalid file ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        '''
        컬럼 타입 조회
        '''
        column_type = ConcatColumnModel.COLUMN_TYPE_CHOICES
        flow_id = request.GET.get("flow_id")

        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not ConcatColumnModel.objects.filter(flow=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        context = {}

        for i in column_type:
            # QuerySet을 list로 변환
            context[i[0]] = list(ConcatColumnModel.objects.filter(
                flow=flow_id,
                column_type=i[0]).values_list('column_name', flat=True))

        return Response(context, status=200)

    @swagger_auto_schema(
        operation_description="concat된 csv파일의 column_type 속성 수정",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'flow_id': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Flow ID"
                ),
                'column_name': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="수정하려는 컬럼의 이름"
                ),
                'column_type': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    enum=["numerical", "categorical", "text",
                          "unavailable"],  # 선택 가능한 옵션
                    description="컬럼의 새로운 타입 (선택 가능)"
                ),
            }
        ),
        request_body_required=True,
        responses={
            200: openapi.Response(description="Column columns updated successfully"),
            400: openapi.Response(description="Invalid request body"),
            404: openapi.Response(description="Column or File not found"),
        },
    )
    def put(self, request, *args, **kwargs):
        '''
        column_type 속성 수정
        '''
        column_name = str(request.data.get("column_name"))
        column_type = str(request.data.get("column_type"))
        flow_id = str(request.data.get("flow_id"))

        if column_name == 'None' or column_type == 'None' or flow_id == 'None':
            return Response(
                {"error": "Both column_name and column_type are required"},
                status=400)

        if not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        if not ConcatColumnModel.objects.filter(flow=flow_id).exists():
            return Response({"error": "File not found"}, status=404)

        column = ConcatColumnModel.objects.filter(
            column_name=column_name, flow=flow_id).first()

        if not column:
            return Response({"error": "Column not found"}, status=404)

        column.column_type = column_type
        column.save()

        return Response(
            {"column_name": column.column_name,
             "column_type": column.column_type},
            status=200)