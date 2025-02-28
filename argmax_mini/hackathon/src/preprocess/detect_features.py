import fireducks.pandas as pd
import numpy as np

def detect_features(df: pd.DataFrame) -> dict:
    """
    각 컬럼을 다양한 기준을 활용하여 자동으로 분류한다.

    분류 기준:
      - numerical: 연속적인 숫자형 데이터
      - categorical: 숫자형이지만 unique 값이 적은 경우 또는 object 타입 데이터
      - numerical_categorical: 숫자형이지만 범주형으로 분류된 경우 (예: unique 값이 20 미만)
      - datetime: 날짜 타입
      - text: 긴 문자열을 포함하는 object 타입

    반환:
      dict: 각 타입별 컬럼 목록과 원본 dtype 정보 포함
    """
    numerical, categorical, numerical_categorical, datetime_cols, text = [], [], [], [], []
    
    dtype_info = {col: str(df[col].dtype) for col in df.columns}  # `.apply(lambda x: x.name).to_dict()` 제거

    for col in df.columns:
        series = df[col]

        # 1. 날짜형 컬럼 탐지 (컬럼명 기반 + datetime 변환)
        if "date" in col.lower() or "time" in col.lower():
            parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            if parsed.notna().mean() >= 0.7:  # 변환된 값이 70% 이상이면 날짜형으로 판단
                datetime_cols.append(col)
                continue

        # 2. 수치형 컬럼 분류 (nunique 최적화)
        if np.issubdtype(series.dtype, np.number):
            unique_vals = series.nunique(dropna=True)
            total_count = len(series)

            if unique_vals < 10 and (unique_vals / total_count < 0.005):
                numerical_categorical.append(col)  # 숫자형이지만 범주형으로 사용됨
            else:
                numerical.append(col)  # 일반 연속형 숫자 데이터
            continue

        # 3. 문자열 데이터 분류 (벡터 연산 적용)
        if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
            avg_length = series.dropna().str.len().mean() if series.dropna().size > 0 else 0
            if avg_length > 50:
                text.append(col)
            else:
                categorical.append(col)
            continue

        # 4. 기본적으로 범주형으로 분류
        categorical.append(col)

    feature_info = {
        'numerical': numerical,
        'categorical': categorical,
        'numerical_categorical': numerical_categorical,
        'datetime': datetime_cols,
        'text': text,
        'dtypes': dtype_info
    }
    
    return feature_info
