# test_analytics.py
import os
import json

from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework.response import Response

import numpy as np
import fireducks.pandas as pd

from data_processing.models import ProjectModel, FlowModel, ConcatColumnModel, ColumnModel, CsvModel, HistogramModel
from data_processing.serializers import ConcatColumnModelSerializer


class HistogramDataTests(APITestCase):
    """
    data_processing 앱의 히스토그램 데이터 엔드포인트를 테스트하는 클래스.
    """

    def setUp(self):
        """
        테스트 환경을 설정합니다.
        - csv 파일 업로드를 위한 URL과 히스토그램 데이터를 가져오는 URL을 설정합니다.
        - ./features 폴더에 있는 csv 파일들을 읽어옵니다.
        """
        self.upload_url = reverse(
            'data_processing:concat-columns-properties')
        self.histogram_url = reverse('data_processing:histograms')
        self.histogram_all_url = reverse('data_processing:histograms-all')
        self.features_dir = os.path.join(
            os.path.dirname(__file__), 'features')
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

            concat_column_serializer = ConcatColumnModelSerializer(data={
                'flow': self.flow.id,
                'column_name': column_name,
                'column_type': column_type,
                'property_type': 'environmental',
                'missing_values_ratio': missing_values_ratio,
            })
            self.columns.append(column_name)

            if concat_column_serializer.is_valid():
                concat_column_serializer.save()
            else:
                return Response(concat_column_serializer.errors, status=400)

            if column_type == "numerical":
                counts, bin_edges = np.histogram(
                    df[column_name].dropna(), bins=10)
                HistogramModel.objects.create(
                    column=concat_column_serializer.instance,
                    counts=json.dumps(counts.tolist()),
                    bin_edges=json.dumps(bin_edges.tolist())
                )
            elif column_type == "categorical":
                # 카테고리별 빈도 계산
                value_counts = df[column_name].dropna().value_counts()
                category_counts = value_counts.tolist()
                category_names = value_counts.index.tolist()

                # 히스토그램 데이터 저장
                HistogramModel.objects.create(
                    column=concat_column_serializer.instance,
                    counts=json.dumps(category_counts),  # 빈도 직렬화
                    bin_edges=json.dumps(category_names)  # 카테고리 이름 직렬화
                )

    def _determine_column_type(self, column):
        """
        컬럼의 데이터 타입을 결정하는 헬퍼 메서드
        """
        if pd.api.types.is_numeric_dtype(column):
            return "numerical"
        elif pd.api.types.is_string_dtype(column):
            return "categorical"
        return "unavailable"

    def test_get_histogram_success(self):
        """
        유효한 flow_id와 column_name으로 히스토그램 데이터를 성공적으로 가져오는지 테스트.
        """
        response = self.client.get(
            self.histogram_url, {'flow_id': self.flow.id, 'column_name': self.columns[0]})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('histograms', response.json())

    def test_get_histogram_no_file_id(self):
        """
        flow_id 없이 히스토그램 데이터를 요청했을 때의 동작을 테스트.
        """
        response = self.client.get(self.histogram_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()['error'],
                         'No flow_id provided')

    def test_get_histogram_file_not_found(self):
        """
        존재하지 않는 flow_id로 히스토그램 데이터를 요청했을 때의 동작을 테스트.
        """
        response = self.client.get(
            self.histogram_url, {'flow_id': '999', 'column_name': self.columns[0]})
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.json()['error'], 'File not found')

    def test_get_histogram_column_not_found(self):
        """
        존재하지 않는 column_name으로 히스토그램 데이터를 요청했을 때의 동작을 테스트.
        """
        response = self.client.get(
            self.histogram_url, {'flow_id': self.flow.id, 'column_name': 'nonexistent_column'})
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.json()['error'], 'File not found')

    def test_get_histogram_all_success(self):
        """
        유효한 flow_id로 모든 컬럼의 히스토그램 데이터를 성공적으로 가져오는지 테스트.
        """
        response = self.client.get(
            self.histogram_all_url, {'flow_id': self.flow.id})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('histograms', response.json())

    def test_get_histogram_all_no_file_id(self):
        """
        flow_id 없이 모든 컬럼의 히스토그램 데이터를 요청했을 때의 동작을 테스트.
        """
        response = self.client.get(self.histogram_all_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json()['error'],
                         'No flow_id provided')

    def test_get_histogram_all_file_not_found(self):
        """
        존재하지 않는 flow_id로 모든 컬럼의 히스토그램 데이터를 요청했을 때의 동작을 테스트.
        """
        response = self.client.get(
            self.histogram_all_url, {'flow_id': '999'})
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.json()['error'], 'File not found')
