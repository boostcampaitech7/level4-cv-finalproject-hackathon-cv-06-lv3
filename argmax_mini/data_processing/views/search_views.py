from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from data_processing.models import SearchResultModel, ConcatColumnModel
from data_processing.serializers import SearchResultModelSerializer


class SearchResultView(APIView):
    '''
    Search Result 조회
    '''
    @swagger_auto_schema(
        operation_description="Search Result 조회",
        manual_parameters=[
            openapi.Parameter(
                'flow_id', openapi.IN_QUERY, type=openapi.TYPE_INTEGER,
                description="ID of the flow",
            ),
        ],
        responses={
            200: openapi.Response(description="Search Result retrieved successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def get(self, request, *args, **kwargs):
        """
        Search Result 조회
        """

        flow_id = request.GET.get("flow_id")
        if not flow_id or not flow_id.isdigit():
            return Response({"error": "No flow_id provided"}, status=400)

        search_results = SearchResultModel.objects.filter(flow=flow_id)
        if not search_results.exists():
            return Response({"error": "File not found"}, status=404)

        serialized_results = SearchResultModelSerializer(
            search_results, many=True).data

        column_ids = {item["column"] for item in serialized_results}

        columns = ConcatColumnModel.objects.filter(id__in=column_ids)
        column_mapping = {column.id: column for column in columns}

        enhanced_results = []
        for item in serialized_results:
            column = column_mapping.get(item["column"])
            enhanced_results.append({
                "column_name": column.column_name if column else None,
                "column_type": column.column_type if column else None,
                "property_type": column.property_type if column else None,
                "ground_truth": item["ground_truth"],
                "predicted": item["predicted"],
                "average_change_rate": item["average_change_rate"],
            })

        return Response({"search_result": enhanced_results}, status=200)
