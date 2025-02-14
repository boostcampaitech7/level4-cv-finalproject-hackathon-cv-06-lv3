from rest_framework.views import APIView
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from data_processing.models import ProjectModel
from data_processing.serializers import ProjectModelSerializer

# 헬퍼 함수 추가: 중복되는 로직을 제거하고 재사용성을 높임


def get_project_by_id(project_id):
    if not project_id or not str(project_id).isdigit():
        raise ValueError("Invalid project_id")
    try:
        return ProjectModel.objects.get(id=project_id)
    except ProjectModel.DoesNotExist:
        raise ProjectModel.DoesNotExist


class ProjectView(APIView):
    """
    프로젝트 생성, 조회, 삭제, 수정 API
    """
    parser_classes = [JSONParser]  # 기본적으로 JSON 파싱을 지원

    @swagger_auto_schema(
        operation_description="프로젝트 생성",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'name': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="프로젝트 이름"
                ),
                'description': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="프로젝트 설명"
                ),
            }
        ),
        request_body_required=True,
        responses={
            201: openapi.Response(description="Project created successfully"),
            400: openapi.Response(description="Invalid request body"),
        },
    )
    def post(self, request, *args, **kwargs):
        """
        프로젝트 생성
        """
        project_serializer = ProjectModelSerializer(data=request.data)
        if project_serializer.is_valid():
            project = project_serializer.save()
            return Response(
                {'project_id': project.id},
                status=status.HTTP_201_CREATED
            )
        return Response(project_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="프로젝트 목록 조회",
        responses={
            200: openapi.Response(description="Projects retrieved successfully"),
        },
    )
    def get(self, request, *args, **kwargs):
        """
        프로젝트 목록 조회
        """
        projects = ProjectModel.objects.all()
        project_serializer = ProjectModelSerializer(projects, many=True)
        return Response({"projects": project_serializer.data}, status=status.HTTP_200_OK)

    @swagger_auto_schema(
        operation_description="프로젝트 삭제",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'project_id': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="ID of the project to delete"
                ),
            }
        ),
        request_body_required=True,
        responses={
            204: openapi.Response(description="Project deleted successfully"),
            400: openapi.Response(description="Invalid project ID"),
            404: openapi.Response(description="Project not found"),
        },
    )
    def delete(self, request, *args, **kwargs):
        """
        프로젝트 삭제
        """
        project_id = request.data.get("project_id")
        try:
            project = get_project_by_id(project_id)
            project.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except ValueError:
            return Response({"error": "Invalid project_id"}, status=status.HTTP_400_BAD_REQUEST)
        except ProjectModel.DoesNotExist:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)

    @swagger_auto_schema(
        operation_description="프로젝트 수정",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'project_id': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="ID of the project to update"
                ),
                'name': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="프로젝트 이름"
                ),
                'description': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="프로젝트 설명"
                ),
            }
        ),
        request_body_required=True,
        responses={
            200: openapi.Response(description="Project updated successfully"),
            400: openapi.Response(description="Invalid request body"),
            404: openapi.Response(description="Project not found"),
        },
    )
    def put(self, request, *args, **kwargs):
        """
        프로젝트 수정
        """
        project_id = request.data.get("project_id")
        try:
            project = get_project_by_id(project_id)
            project_serializer = ProjectModelSerializer(
                project, data=request.data, partial=True
            )
            if project_serializer.is_valid():
                project = project_serializer.save()
                return Response(
                    {'project_id': project.id},
                    status=status.HTTP_200_OK
                )
            return Response(project_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            return Response({"error": "Invalid project_id"}, status=status.HTTP_400_BAD_REQUEST)
        except ProjectModel.DoesNotExist:
            return Response({"error": "Project not found"}, status=status.HTTP_404_NOT_FOUND)
