# test_csv.py
import io
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
import fireducks.pandas as pd
from data_processing.models import CsvModel, ProjectModel


class CsvViewTests(APITestCase):
    """
    CsvView 클래스의 엔드포인트를 테스트합니다.
    """

    def setUp(self):
        """
        테스트 환경 설정
        """
        self.base_url = reverse('data_processing:csvs')  # URL 네임스페이스에 맞게 수정

        # 테스트용 프로젝트 생성
        self.project = ProjectModel.objects.create(
            name="Test Project", description="Test Description"
        )
        self.writer = "test_writer"

        # 테스트용 CSV 파일 생성
        self.csv_file = io.BytesIO()
        df = pd.DataFrame({
            'column1': [1, 2, 3, 4, 5],
            'column2': ['A', 'B', 'A', 'B', 'C']
        })
        df.to_csv(self.csv_file, index=False)
        self.csv_file.seek(0)  # 파일 포인터를 처음으로 이동

    def test_post_single_csv_model_file(self):
        """
        단일 CsvModel 파일 업로드 테스트
        """
        response = self.client.post(
            self.base_url,
            {
                'csv_file': self.csv_file,  # 파일 필드 이름을 'csv_file'로 변경
                'writer': self.writer,
                'project_id': self.project.id
            },
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('file_id', response.json())

    def test_post_invalid_csv_file(self):
        """
        잘못된 CSV 파일 업로드 테스트
        """
        invalid_csv_file = io.BytesIO(b"invalid,data\n1,2,3\n4,5,6")
        response = self.client.post(
            self.base_url,
            {
                'csv_file1': invalid_csv_file,  # 파일 필드 이름을 'csv_file'로 변경
                'writer': self.writer,
                'project_id': self.project.id
            },
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.json())

    def test_post_empty_csv_file(self):
        """
        빈 CSV 파일 업로드 테스트
        """
        empty_csv_file = io.BytesIO()
        response = self.client.post(
            self.base_url,
            {
                'csv_file': empty_csv_file,  # 파일 필드 이름을 'csv_file'로 변경
                'writer': self.writer,
                'project_id': self.project.id
            },
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.json())

    def test_post_missing_required_fields(self):
        """
        필수 필드 누락 시 업로드 테스트
        """
        # writer 누락
        response = self.client.post(
            self.base_url,
            {
                'file': self.csv_file,
                'project_id': self.project.id
            },
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.json())

        # project_id 누락
        response = self.client.post(
            self.base_url,
            {
                'file': self.csv_file,
                'writer': self.writer
            },
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.json())

    def test_get_csv_model_file_list(self):
        """
        업로드된 CsvModel 파일 목록 조회 테스트
        """
        # 파일 업로드
        self.client.post(
            self.base_url,
            {
                'csv_file': self.csv_file,  # 파일 필드 이름을 'csv_file'로 변경
                'writer': self.writer,
                'project_id': self.project.id
            },
            format='multipart'
        )

        # 파일 목록 조회
        response = self.client.get(self.base_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('csvs', response.json())

    def test_delete_csv_model_file(self):
        """
        업로드된 CsvModel 파일 삭제 테스트
        """
        # 파일 업로드
        response = self.client.post(
            self.base_url,
            {
                'csv_file': self.csv_file,  # 파일 필드 이름을 'csv_file'로 변경
                'writer': self.writer,
                'project_id': self.project.id
            },
            format='multipart'
        )
        file_id = response.json()['file_id']

        # 파일 삭제
        response = self.client.delete(
            self.base_url,
            data={'file_id': file_id},
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        # 파일이 삭제되었는지 확인
        self.assertFalse(CsvModel.objects.filter(id=file_id).exists())

    def test_delete_nonexistent_csv_model_file(self):
        """
        존재하지 않는 CsvModel 파일 삭제 시도 테스트
        """
        response = self.client.delete(
            self.base_url,
            data={'file_id': 999},  # 존재하지 않는 파일 ID
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn('error', response.json())
        self.assertEqual(response.json()['error'], 'File not found')

    def test_delete_invalid_file_id(self):
        """
        유효하지 않은 file_id로 삭제 시도 테스트
        """
        response = self.client.delete(
            self.base_url,
            data={'file_id': 'invalid_id'},  # 유효하지 않은 파일 ID
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.json())
        self.assertEqual(response.json()['error'],
                         'Invalid or missing file_id')
