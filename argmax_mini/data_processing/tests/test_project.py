import json
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from data_processing.models import ProjectModel


class ProjectViewTests(APITestCase):
    """
    ProjectView 클래스의 엔드포인트를 테스트합니다.
    - 프로젝트 생성
    - 프로젝트 목록 조회
    - 프로젝트 삭제
    - 프로젝트 수정
    """

    def setUp(self):
        """
        테스트 환경을 설정합니다.
        """
        self.base_url = reverse('data_processing:projects')

        # 테스트용 프로젝트 생성
        self.project1 = ProjectModel.objects.create(
            name="Test Project 1", description="Description 1")
        self.project2 = ProjectModel.objects.create(
            name="Test Project 2", description="Description 2")

    def test_create_project(self):
        """
        프로젝트 생성 테스트
        """
        data = {
            "name": "New Project",
            "description": "This is a new project."
        }
        response = self.client.post(self.base_url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("project_id", response.json())

        # DB 확인
        project_id = response.json()["project_id"]
        self.assertTrue(ProjectModel.objects.filter(id=project_id).exists())

    def test_create_project_invalid(self):
        """
        잘못된 데이터로 프로젝트 생성 시도 테스트
        """
        data = {"description": "Missing name"}  # 이름이 없음
        response = self.client.post(self.base_url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("name", response.json())

    def test_get_project_list(self):
        """
        프로젝트 목록 조회 테스트
        """
        response = self.client.get(self.base_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("projects", response.json())

        # 응답에 프로젝트가 포함되었는지 확인
        projects = response.json()["projects"]
        self.assertEqual(len(projects), 2)

    def test_delete_project(self):
        """
        프로젝트 삭제 테스트
        """
        response = self.client.delete(
            self.base_url, {'project_id': self.project1.id}, format='json')

        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        # DB 확인
        self.assertFalse(ProjectModel.objects.filter(id=self.project1.id).exists())

    def test_delete_project_not_found(self):
        """
        존재하지 않는 프로젝트 삭제 시도 테스트
        """
        response = self.client.delete(
            self.base_url, {'project_id': 9999}, format='json')

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Project not found")

    def test_update_project(self):
        """
        프로젝트 수정 테스트
        """
        data = {
            "project_id": self.project1.id,
            "name": "Updated Project",
            "description": "Updated description."
        }
        response = self.client.put(self.base_url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("project_id", response.json())

        # DB 확인
        project = ProjectModel.objects.get(id=self.project1.id)
        self.assertEqual(project.name, "Updated Project")
        self.assertEqual(project.description, "Updated description.")

    def test_update_project_partial(self):
        """
        프로젝트 부분 수정 테스트
        """
        data = {
            "project_id": self.project1.id,
            "description": "Partially updated description."
        }
        response = self.client.put(self.base_url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("project_id", response.json())

        # DB 확인
        project = ProjectModel.objects.get(id=self.project1.id)
        self.assertEqual(project.description, "Partially updated description.")

    def test_update_project_not_found(self):
        """
        존재하지 않는 프로젝트 수정 시도 테스트
        """
        data = {
            "project_id": 9999,
            "name": "Nonexistent Project"
        }
        response = self.client.put(self.base_url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Project not found")
