# E-Commerce 데이터 전처리 모듈

## 개요

이 모듈은 E-Commerce 데이터셋을 전처리하는 파이프라인을 포함하고 있습니다.
주어진 Parquet 파일들을 로드하고, 데이터 변환, 병합, 피처 엔지니어링을 수행하여 최종적으로 가공된 데이터를 저장합니다.

## 파일 구조

```
ecommerce/
│── __init__.py
│── data_loading.py
│── data_saving.py
│── data_transformation.py
│── feature_engineering.py
│── pipeline.py
└── README.md
```

### 1. `__init__.py`

- `ecommerce` 폴더를 Python 패키지로 인식할 수 있도록 하는 초기화 파일입니다.
- 데이터 로딩, 병합, 저장, 피처 엔지니어링 관련 함수를 모듈에서 쉽게 불러올 수 있도록 설정되어 있습니다.

### 2. `data_loading.py`

- 여러 개의 Parquet 파일을 `polars.DataFrame`으로 로드하는 기능을 포함합니다.

- ```
  load_data()
  ```

   함수는 다음과 같은 원본 데이터를 로드합니다.

  - `brand_table.parquet`
  - `category_table.parquet`
  - `item_data.parquet`
  - `log_data.parquet`
  - `user_data.parquet`

### 3. `data_saving.py`

- DataFrame을 Parquet 파일로 저장하는 기능을 포함합니다.
- `save_parquet(df, file_path)`: DataFrame을 지정된 경로에 Parquet 형식으로 저장합니다.
- `split_and_save(df, chunk_size, output_dir, file_prefix)`: `product_id_index`를 기준으로 데이터를 청크로 나누어 저장합니다.

### 4. `data_transformation.py`

- 데이터를 병합하고 필요한 열을 선택하여 가공하는 기능을 수행합니다.

- ```
  merge_initial_data(brand_df, category_df, item_df, log_df, user_df)
  ```
  - 브랜드 데이터와 아이템 데이터를 `brand_id` 기준으로 병합
  - 카테고리 데이터를 `category_id` 기준으로 병합
  - 로그 데이터와 사용자 데이터를 `user_session_index` 기준으로 병합
  - `category_2_id`, `category_3_id`를 결합하여 간소화된 `category_id` 생성
  - 최종 병합된 원본 데이터 반환

### 5. `feature_engineering.py`

- 제품별 할인율, 구매율, 판매량 등을 계산하여 피처 엔지니어링을 수행합니다.

- ```
  perform_feature_engineering(chunk_df)
  ```
  - `regular_price`, `discount_rate` 계산
  - 이상치 제거 (price ≤ 0)
  - `event_type_index` 기준으로 조회(View), 구매(Purchase) 이벤트 집계 후 구매율(`purchase_rate`) 계산
  - 판매량(`sales_volume`) 및 매출(`revenue`) 추정
  - 제품별 할인율 통계(`discount_rate`) 산출 및 필터링
  - 불필요한 열 제거 후 최종 가공된 데이터 반환

### 6. `pipeline.py`

- 전체 데이터 전처리 과정을 실행하는 파이프라인 스크립트입니다.
- 실행 과정:
  1. 원본 데이터 로드 (`load_data()`)
  2. 데이터 병합 (`merge_initial_data()`)
  3. `product_id_index` 기준으로 데이터를 청크 단위로 저장 (`split_and_save()`)
  4. 각 청크에 대해 피처 엔지니어링 수행 (`perform_feature_engineering()`)
  5. 모든 가공된 청크를 병합하여 최종 데이터 저장 (`final_engineered_data.parquet`)

## 데이터 파일 다운로드 및 경로

해당 모듈에서 사용하는 데이터 파일은 아래 링크에서 다운로드할 수 있습니다:

🔗 **[E-Commerce 데이터 다운로드](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)**

### 데이터 다운로드 후 저장 경로

다운로드한 Parquet 파일들은 다음 경로에 저장하여 사용하면 됩니다:

```
/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/argmax_mini/hackathon/data/raw
```

### 필요한 데이터 파일 목록

- `brand_table.parquet`
- `category_table.parquet`
- `item_data.parquet`
- `log_data.parquet`
- `user_data.parquet`

## 실행 방법

```
python pipeline.py
```

- 실행하면 데이터 로드, 변환, 병합, 피처 엔지니어링을 수행한 후 최종 Parquet 파일이 생성됩니다.