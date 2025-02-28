# Preprocess Module

`preprocess` 폴더는 데이터 전처리를 담당하는 다양한 스크립트로 구성되어 있으며, 데이터 정제, 스케일링, 이상치 제거, 인코딩 등의 기능을 포함합니다. 본 문서는 각 파일의 기능과 사용 방법을 설명합니다.

## Directory Structure

```
hackathon/
│── preprocess_main.py
│── src/
│   │── preprocess/
│   │   │── __init__.py
│   │   │── analyze_distribution.py
│   │   │── datetime_features.py
│   │   │── detect_features.py
│   │   │── dynamic_encode.py
│   │   │── dynamic_outlier.py
│   │   │── dynamic_scaling.py
│   │   │── encoding.py
│   │   │── missing_values.py
│   │   │── outlier_detection.py
│   │   │── processing_metadata.py
│   │   │── sampling.py
│   │   │── scaling.py
│   │   │── text_processing.py
│   │── dynamic_pipeline.py
```

## Modules Overview

### 1. `analyze_distribution.py`

수치형 데이터의 분포를 분석하고, 정규 분포 여부를 판단하는 기능을 제공합니다.

- `**analyze_distribution(df: pd.DataFrame, numerical_cols: list) -> dict**`: 수치형 컬럼의 왜도(skewness)와 Shapiro-Wilk 테스트 결과를 반환하여 정규 분포 여부를 판단합니다.

### 2. `datetime_features.py`

날짜형 데이터를 처리하는 기능을 포함합니다.

- `**remove_datetime_columns(df: pd.DataFrame, datetime_columns: list) -> pd.DataFrame**`: 특정 날짜 컬럼을 제거하고, 해당 컬럼을 메타데이터에 기록합니다.

### 3. `detect_features.py`

데이터프레임 내 각 컬럼을 자동으로 유형(numerical, categorical, datetime, text 등) 분류합니다.

- `**detect_features(df: pd.DataFrame) -> dict**`: 다양한 기준을 활용하여 컬럼을 분석하고 적절한 타입으로 분류합니다.

### 4. `dynamic_encode.py`

BERT 임베딩과 Label Encoding을 활용하여 범주형 데이터를 인코딩합니다.

- `**dynamic_encode(df: pd.DataFrame, feature_info: dict, scaler: dict) -> pd.DataFrame**`: BERT 기반 텍스트 임베딩과 Label Encoding을 적용합니다.

### 5. `dynamic_outlier.py`

데이터의 분포를 기반으로 이상치를 동적으로 제거하는 기능을 제공합니다.

- `**dynamic_outlier_removal(df: pd.DataFrame, numerical_cols: list, \**kwargs) -> pd.DataFrame**`: 정규 분포 여부에 따라 Z-score 또는 IQR 방식을 사용하여 이상치를 제거합니다.

### 6. `dynamic_scaling.py`

각 컬럼의 분포를 분석하여 적절한 스케일링 기법(StandardScaler, MinMaxScaler, RobustScaler)을 자동 선택합니다.

- `**dynamic_scaling(df: pd.DataFrame, numerical_cols: list, scalers: dict) -> pd.DataFrame**`: 정규 분포 여부에 따라 적절한 스케일링 방법을 적용합니다.

### 7. `encoding.py`

범주형 데이터에 대한 다양한 인코딩 방법을 제공합니다.

- `**one_hot_encode(df: pd.DataFrame, cols: list) -> pd.DataFrame**`: One-Hot Encoding을 적용합니다.
- `**label_encode(df: pd.DataFrame, cols: list, scaler: dict) -> pd.DataFrame**`: Label Encoding을 수행하고, 인코더를 저장합니다.
- `**bert_encode(categories: list, model, tokenizer) -> pd.DataFrame**`: BERT 임베딩을 활용하여 텍스트 데이터를 벡터화합니다.

### 8. `missing_values.py`

결측치 처리를 위한 다양한 방법을 제공합니다.

- `**fill_missing_numerical(df: pd.DataFrame, numerical_cols: list, strategy: str = 'median') -> pd.DataFrame**`: 수치형 컬럼의 결측치를 평균 또는 중앙값으로 대체합니다.
- `**fill_missing_categorical(df: pd.DataFrame, categorical_cols: list, fill_value: str = 'Unknown') -> pd.DataFrame**`: 범주형 컬럼의 결측치를 특정 값으로 대체합니다.
- `**drop_high_missing_data(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame**`: 결측치 비율이 높은 컬럼을 제거합니다.

### 9. `outlier_detection.py`

이상치 제거를 위한 다양한 기법을 제공합니다.

- `**remove_outliers_iqr(df: pd.DataFrame, numerical_cols: list, factor: float = 1.5) -> pd.DataFrame**`: IQR 방식으로 이상치를 제거합니다.
- `**remove_outliers_zscore(df: pd.DataFrame, numerical_cols: list, threshold: float = 3.0) -> pd.DataFrame**`: Z-score 방식으로 이상치를 제거합니다.

### 10. `processing_metadata.py`

전처리 과정에서 제거된 컬럼을 기록하고 관리합니다.

- `**add_removed_columns(columns: list)**`: 제거된 컬럼을 기록합니다.
- `**get_removed_columns() -> list**`: 제거된 컬럼 목록을 반환합니다.
- `**reset_metadata()**`: 메타데이터를 초기화합니다.

### 11. `sampling.py`

데이터 샘플링을 수행하는 기능을 포함합니다.

- `**sample_dataframe(df: pd.DataFrame, sample_size: int = 1_000_000, random_state: int = 42) -> pd.DataFrame**`: 랜덤 샘플링을 수행하여 일부 데이터만 추출합니다.

### 12. `scaling.py`

수치형 데이터를 스케일링하는 다양한 방법을 제공합니다.

- `**scale_data(df: pd.DataFrame, numerical_cols: list, method: str = 'standard') -> (pd.DataFrame, object)**`: 지정된 스케일링 방식(Standard, MinMax, Robust)으로 데이터를 변환합니다.

### 13. `text_processing.py`

텍스트 데이터를 전처리하는 기능을 제공합니다.

- `**process_text(df: pd.DataFrame, text_columns: list) -> pd.DataFrame**`: 텍스트를 소문자로 변환하고 특수문자를 제거합니다


### 14.dynamic_pipeline.py

데이터 전처리를 자동화하는 주요 파이프라인입니다.

- `**preprocess_dynamic(df: pd.DataFrame) -> pd.DataFrame**`:
  - 샘플링을 통해 컬럼을 분류합니다.
  - 결측치, 이상치 및 스케일링 처리를 수행합니다.
  - 텍스트 및 날짜형 데이터를 정리합니다.
  - Label Encoding과 BERT 기반 인코딩을 동적으로 적용합니다.
  - 최종적으로 전처리가 완료된 데이터프레임을 반환합니다.



## Preprocessing Execution

### `preprocess_main.py`

### 설명

이 파일은 사용자가 입력한 CSV 또는 Parquet 데이터를 불러와 전처리 파이프라인(`preprocess_dynamic`)을 실행하고, 전처리된 결과를 저장하는 역할을 합니다.

### 실행 방법

```
python preprocess_main.py /path/to/data.csv
python preprocess_main.py /path/to/data.parquet
```

### 주요 기능

- 입력 데이터의 형식(CSV 또는 Parquet)을 감지하여 로드
- `preprocess_dynamic()`을 실행하여 동적 전처리 수행
- 전처리된 데이터를 원본 확장자 형식을 유지한 상태로 저장