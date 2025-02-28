import os
import json

from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APITestCase
import fireducks.pandas as pd

from data_processing.models import ProjectModel, FlowModel, ConcatColumnModel, ColumnModel, CsvModel


class ColumnViewTests(APITestCase):
    def setUp(self):
        """
        테스트를 위한 데이터 초기화
        """
        self.columns_url = reverse('data_processing:columns')
        self.features_dir = os.path.join(
            os.path.dirname(__file__), './features')

        # CSV 파일 로드 및 데이터베이스에 삽입
        self.csvs = [f for f in os.listdir(
            self.features_dir) if f.endswith('.csv')]
        if not self.csvs:
            raise FileNotFoundError(
                "No csv files found in ./features directory.")

        file_path = os.path.join(self.features_dir, self.csvs[0])
        df = pd.read_csv(file_path)

        # Project 생성
        self.project = ProjectModel.objects.create(
            name="test_project",
            description="test_description"
        )

        # CSV 파일 업로드 및 CsvModel 생성
        csv_file = SimpleUploadedFile(
            self.csvs[0], df.to_csv(index=False).encode(), content_type="text/csv"
        )
        self.csv = CsvModel.objects.create(
            project=self.project,
            csv=csv_file,
            writer="test_writer",
            size=os.path.getsize(file_path),
            rows=len(df)
        )

        # ColumnModel 생성 (CSV 파일의 컬럼명 저장)
        self.columns = []
        for column_name in df.columns:
            column_record = ColumnModel.objects.create(
                csv=self.csv,
                column_name=column_name
            )
            self.columns.append(column_record)

    def test_get_column_names_success(self):
        """
        CSV 파일의 컬럼명을 성공적으로 조회하는지 테스트
        """
        response = self.client.get(self.columns_url, {"csv_id": self.csv.id})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("column_names", response.json())

        # 응답 데이터의 컬럼명과 데이터베이스의 컬럼명 비교
        expected_column_names = list(ColumnModel.objects.filter(
            csv=self.csv).values_list('column_name', flat=True))
        self.assertEqual(
            response.json()["column_names"], expected_column_names)

    def test_get_column_names_invalid_csv_id(self):
        """
        유효하지 않은 csv_id로 요청을 보내는 경우를 테스트
        """
        response = self.client.get(self.columns_url, {"csv_id": 9999})
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "File not found")

    def test_get_column_names_no_csv_id(self):
        """
        쿼리파라미터로 csv_id가 전달되지 않은 경우를 테스트
        """
        response = self.client.get(self.columns_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "No csv_id provided")

    def test_get_column_names_non_digit_csv_id(self):
        """
        csv_id가 숫자가 아닌 경우를 테스트
        """
        response = self.client.get(self.columns_url, {"csv_id": "invalid"})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "No csv_id provided")


class ConcatColumnPropertiesViewTests(APITestCase):
    def setUp(self):
        """
        테스트를 위한 데이터 초기화
        """
        self.columns_url = reverse(
            'data_processing:concat-columns-properties')
        self.features_dir = os.path.join(
            os.path.dirname(__file__), './features')

        # CSV 파일 로드 및 데이터베이스에 삽입
        self.csvs = [f for f in os.listdir(
            self.features_dir) if f.endswith('.csv')]
        if not self.csvs:
            raise FileNotFoundError(
                "No csv files found in ./features directory.")

        file_path = os.path.join(self.features_dir, self.csvs[0])
        df = pd.read_csv(file_path)

        # Project 생성
        self.project = ProjectModel.objects.create(
            name="test_project",
            description="test_description"
        )

        # CSV 파일 업로드 및 CsvModel 생성
        csv_file = SimpleUploadedFile(
            self.csvs[0], df.to_csv(index=False).encode(), content_type="text/csv"
        )
        self.csv = CsvModel.objects.create(
            project=self.project,
            csv=csv_file,
            writer="test_writer",
            size=os.path.getsize(file_path),
            rows=len(df)
        )

        # FlowModel 생성
        self.flow = FlowModel.objects.create(
            project=self.project,
            flow_name="test_flow",
            concat_csv=csv_file
        )
        self.flow.csv.add(self.csv)

        # ConcatColumn 생성
        self.columns = []
        for column_name in df.columns:
            column_type = self._determine_column_type(df[column_name])
            missing_values_ratio = round(
                df[column_name].isnull().mean() * 100, 2)
            if missing_values_ratio > 50:
                column_type = "unavailable"

            column_record = ConcatColumnModel.objects.create(
                flow=self.flow,
                column_name=column_name,
                column_type=column_type,
                missing_values_ratio=missing_values_ratio
            )
            self.columns.append(column_record)

    def _determine_column_type(self, column):
        """
        컬럼의 데이터 타입을 결정하는 헬퍼 메서드
        """
        if pd.api.types.is_numeric_dtype(column):
            return "numerical"
        elif pd.api.types.is_string_dtype(column):
            if 'address' in column.name.lower() or 'description' in column.name.lower() or 'text' in column.name.lower():
                return "text"
            return "categorical"
        return "unavailable"

    def test_put_property_type_success(self):
        """
        컬럼 타입을 성공적으로 업데이트하는지 테스트
        """
        column_name = self.columns[0].column_name
        new_property_type = "Controllable"

        data = {
            "flow_id": self.flow.id,
            "column_name": column_name,
            "property_type": new_property_type
        }

        response = self.client.put(self.columns_url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["column_name"], column_name)
        self.assertEqual(response.json()["property_type"], new_property_type)

    def test_put_property_type_invalid_request(self):
        """
        잘못된 요청 데이터로 컬럼 타입 업데이트를 테스트
        """
        data = {
            "flow_id": self.flow.id,
            "column_name": None,
            "property_type": None
        }

        response = self.client.put(self.columns_url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            response.json()["error"],
            "Both column_name and property_type are required"
        )

    def test_put_property_type_not_found(self):
        """
        존재하지 않는 컬럼 이름으로 요청을 보내는 경우를 테스트
        """
        data = {
            "flow_id": self.flow.id,
            "column_name": "not_found_column",
            "property_type": "environmental"
        }

        response = self.client.put(self.columns_url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.json()["error"], "Column not found")

    def test_put_invalid_flow_id(self):
        """
        유효하지 않은 flow_id로 요청을 보내는 경우를 테스트
        """
        data = {
            "flow_id": 9999,
            "column_name": self.columns[0].column_name,
            "property_type": "controllable"
        }

        response = self.client.put(self.columns_url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.json()["error"], "File not found")

    def test_get_column_list(self):
        """
        컬럼 리스트를 성공적으로 가져오는지 테스트
        """
        response = self.client.get(
            self.columns_url, {"flow_id": self.flow.id})
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # 컬럼 타입별로 컬럼 이름 리스트 확인
        self.assertIn("numerical", response.json())
        self.assertIn("categorical", response.json())
        self.assertIn("unavailable", response.json())

    def test_get_column_list_invalid_flow_id(self):
        """
        유효하지 않은 flow_id로 컬럼 리스트를 가져오는 경우를 테스트
        """
        response = self.client.get(self.columns_url, {"flow_id": 9999})
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.json()["error"], "File not found")

    def test_get_column_list_no_flow_id(self):
        """
        쿼리파라미터로 flow_id가 전달되지 않은 경우를 테스트
        """
        response = self.client.get(self.columns_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()["error"], "No flow_id provided")
