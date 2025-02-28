from django.urls import reverse
from rest_framework.test import APITestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from data_processing.models import ProjectModel, FlowModel, CsvModel


class FlowsViewTests(APITestCase):
    """
    FlowsView 클래스의 엔드포인트를 테스트합니다.
    """

    def setUp(self):
        """
        테스트 환경 설정
        """
        # 테스트용 프로젝트 생성
        self.project = ProjectModel.objects.create(
            name="Test Project", description="Test Description"
        )
        # 테스트용 Flow 생성
        self.flow = FlowModel.objects.create(
            project=self.project, flow_name="Test Flow"
        )

        # URL 네임스페이스에 맞게 수정
        self.base_url = reverse('data_processing:flows')

    def test_get_flows(self):
        """
        프로젝트의 Flow 조회 테스트
        """
        response = self.client.get(
            self.base_url, {"project_id": self.project.id}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("flows", response.data)
        self.assertGreaterEqual(len(response.data["flows"]), 1)

    def test_get_flows_invalid_project_id(self):
        """
        유효하지 않은 project_id로 Flow 조회 시도 테스트
        """
        response = self.client.get(
            self.base_url, {"project_id": "invalid_id"}
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_get_flows_project_not_found(self):
        """
        존재하지 않는 project_id로 Flow 조회 시도 테스트
        """
        response = self.client.get(
            self.base_url, {"project_id": 999}
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)

    def test_create_flow(self):
        """
        프로젝트의 Flow 생성 테스트
        """
        response = self.client.post(
            self.base_url,
            {"project_id": self.project.id, "flow_name": "New Flow"},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("flow_id", response.data)

        # 데이터베이스에 Flow가 생성되었는지 확인
        flow_id = response.data["flow_id"]
        self.assertTrue(FlowModel.objects.filter(id=flow_id).exists())

    def test_create_flow_missing_project_id(self):
        """
        project_id 누락 시 Flow 생성 시도 테스트
        """
        response = self.client.post(
            self.base_url,
            {"flow_name": "New Flow"},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_create_flow_missing_flow_name(self):
        """
        flow_name 누락 시 Flow 생성 시도 테스트
        """
        response = self.client.post(
            self.base_url,
            {"project_id": self.project.id},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_create_flow_project_not_found(self):
        """
        존재하지 않는 project_id로 Flow 생성 시도 테스트
        """
        response = self.client.post(
            self.base_url,
            {"project_id": 999, "flow_name": "New Flow"},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)

    def test_delete_flow(self):
        """
        프로젝트의 Flow 삭제 테스트
        """
        response = self.client.delete(
            self.base_url,
            {"flow_id": self.flow.id},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("flow_id", response.data)

        # 데이터베이스에서 Flow가 삭제되었는지 확인
        self.assertFalse(FlowModel.objects.filter(id=self.flow.id).exists())

    def test_delete_flow_invalid_flow_id(self):
        """
        유효하지 않은 flow_id로 Flow 삭제 시도 테스트
        """
        response = self.client.delete(
            self.base_url,
            {"flow_id": "invalid_id"},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_delete_flow_not_found(self):
        """
        존재하지 않는 flow_id로 Flow 삭제 시도 테스트
        """
        response = self.client.delete(
            self.base_url,
            {"flow_id": 999},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)

    def test_update_flow(self):
        """
        프로젝트의 Flow 수정 테스트
        """
        new_flow_name = "Updated Flow"
        response = self.client.put(
            self.base_url,
            {"flow_id": self.flow.id, "flow_name": new_flow_name},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("flow_id", response.data)
        self.assertIn("flow_name", response.data)

        # 데이터베이스에서 Flow 이름이 업데이트되었는지 확인
        self.flow.refresh_from_db()
        self.assertEqual(self.flow.flow_name, new_flow_name)

    def test_update_flow_missing_flow_id(self):
        """
        flow_id 누락 시 Flow 수정 시도 테스트
        """
        response = self.client.put(
            self.base_url,
            {"flow_name": "Updated Flow"},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_update_flow_missing_flow_name(self):
        """
        flow_name 누락 시 Flow 수정 시도 테스트
        """
        response = self.client.put(
            self.base_url,
            {"flow_id": self.flow.id},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_update_flow_not_found(self):
        """
        존재하지 않는 flow_id로 Flow 수정 시도 테스트
        """
        response = self.client.put(
            self.base_url,
            {"flow_id": 999, "flow_name": "Updated Flow"},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)

class FlowCsvAddViewTests(APITestCase):
    """
    FlowCsvView 클래스의 엔드포인트를 테스트합니다.
    """

    def setUp(self):
        """
        테스트 환경 설정
        """
        # 테스트용 프로젝트 생성
        self.project = ProjectModel.objects.create(
            name="Test Project", description="Test Description"
        )
        # 테스트용 Flow 생성
        self.flow = FlowModel.objects.create(
            project=self.project, flow_name="Test Flow"
        )

        # 테스트용 CSV 생성
        csv_content1 = b'A,B,C\n1,2,3\n4,5,6'
        self.csv_file1 = SimpleUploadedFile(
            "test1.csv", csv_content1, content_type="text/csv"
        )
        self.csv1 = CsvModel.objects.create(
            project=self.project,
            csv=self.csv_file1,
            writer="writer1",
            size=round(len(csv_content1) / 1024, 2),
            rows=2
        )

        csv_content2 = b'A,B,D\n7,8,9\n10,11,12'
        self.csv_file2 = SimpleUploadedFile(
            "test2.csv", csv_content2, content_type="text/csv"
        )
        self.csv2 = CsvModel.objects.create(
            project=self.project,
            csv=self.csv_file2,
            writer="writer2",
            size=round(len(csv_content2) / 1024, 2),
            rows=2
        )
        self.base_url = reverse('data_processing:flow_add_csv')  # URL 네임스페이스에 맞게 수정

    def test_add_csv_to_flow(self):
        """
        Flow에 CSV 추가 테스트
        """
        response = self.client.post(
            self.base_url,
            {"flow_id": self.flow.id, "csv_ids": [self.csv1.id, self.csv2.id]},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("flow_id", response.data)
        self.assertIn("csv_ids", response.data)

        # Flow에 CSV가 추가되었는지 확인
        self.assertEqual(self.flow.csv.count(), 2)

    def test_add_csv_to_flow_invalid_flow_id(self):
        """
        유효하지 않은 flow_id로 CSV 추가 시도 테스트
        """
        response = self.client.post(
            self.base_url,
            {"flow_id": "invalid_id", "csv_ids": [self.csv1.id]},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_add_csv_to_flow_flow_not_found(self):
        """
        존재하지 않는 flow_id로 CSV 추가 시도 테스트
        """
        response = self.client.post(
            self.base_url,
            {"flow_id": 999, "csv_ids": [self.csv1.id]},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)

    def test_add_csv_to_flow_invalid_csv_id(self):
        """
        유효하지 않은 csv_id로 CSV 추가 시도 테스트
        """
        response = self.client.post(
            self.base_url,
            {"flow_id": self.flow.id, "csv_ids": ["invalid_id"]},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_add_csv_to_flow_csv_not_found(self):
        """
        존재하지 않는 csv_id로 CSV 추가 시도 테스트
        """
        response = self.client.post(
            self.base_url,
            {"flow_id": self.flow.id, "csv_ids": [999]},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)