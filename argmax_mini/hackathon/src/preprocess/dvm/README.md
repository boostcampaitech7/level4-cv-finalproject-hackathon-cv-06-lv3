# DVM 데이터 전처리 모듈

## 개요

이 모듈은 DVM 데이터셋을 전처리하는 파이프라인을 포함하고 있습니다.
주어진 CSV 파일들을 로드하고, 데이터 변환, 병합, 피처 엔지니어링을 수행하여 최종적으로 가공된 데이터를 저장합니다.

## 파일 구조

```
dvm/
│── __init__.py
│── data_loading.py
│── data_saving.py
│── data_transformation.py
│── feature_engineering.py
│── pipeline.py
└── README.md
```

### 1. `__init__.py`

- `dvm` 폴더를 Python 패키지로 인식할 수 있도록 하는 초기화 파일입니다.

### 2. `data_loading.py`

- 여러 개의 CSV 파일을 `pandas.DataFrame`으로 로드하는 기능을 포함합니다.
- 파일이 존재하지 않을 경우 에러 메시지를 출력합니다.

### 3. `data_saving.py`

- 최종 가공된 DataFrame을 CSV 파일로 저장하는 기능을 포함합니다.

### 4. `data_transformation.py`

- `Sales_table`을 wide 형식에서 long 형식으로 변환 (`melt_sales_table`)
- `Price_table`과 변환된 `Sales_table`을 병합 (`merge_price_sales`)
- `sales` 값을 특정 기준으로 집계 (`aggregate_sales`)

### 5. `feature_engineering.py`

- `calculate_mode`: 특정 열을 기준으로 그룹화 후 최빈값을 계산
- `merge_ad_tables`: `Ad_table1`, `Ad_table2`를 집계된 데이터와 병합
- `add_annual_revenue`: `sales`와 `Entry_price`를 곱하여 `Annual_revenue` 열을 생성

### 6. `pipeline.py`

- 전체 데이터 전처리 과정이 포함된 실행 스크립트
- CSV 파일 로드 → 데이터 변환 및 병합 → 피처 엔지니어링 → 최종 데이터 저장 순서로 진행
- 실행 시 최종 가공된 데이터가 `dvm_merged_df.csv` 파일로 저장됨

## 실행 방법

```
python pipeline.py
```

- 실행하면 데이터 로드, 변환, 병합, 피처 엔지니어링을 수행한 후 최종 CSV 파일이 생성됩니다.

## 데이터 파일 다운로드 및 경로

해당 모듈에서 사용하는 데이터 파일은 아래 링크에서 다운로드할 수 있습니다:

🔗 **[DVM 데이터 다운로드](https://deepvisualmarketing.github.io/)**

### 데이터 다운로드 후 저장 경로

다운로드한 CSV 파일들은 다음 경로에 저장하여 사용하면 됩니다:

```
/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/argmax_mini/hackathon/data
```

### 필요한 데이터 파일 목록

- `Basic_table.csv`
- `Price_table.csv`
- `Sales_table.csv`
- `Trim_table.csv`
- `Ad_table.csv` (`Ad_table1`)
- `Ad_table (extra).csv` (`Ad_table2`)
- `Image_table.csv`

## 주의 사항

- `pipeline.py` 실행 전에 모든 CSV 파일이 존재하는지 확인해야 합니다.