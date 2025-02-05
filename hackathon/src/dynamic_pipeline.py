import pandas as pd
from src.preprocess.datetime_features import remove_datetime_columns
from src.preprocess.detect_features import detect_features
from src.preprocess.dynamic_encoding import dynamic_encode
from src.preprocess.dynamic_outlier import dynamic_outlier_removal
from src.preprocess.dynamic_scaling import dynamic_scaling
from src.preprocess.geospatial_features import remove_geospatial_columns
from src.preprocess.identity_scaler import IdentityScaler
from src.preprocess.missing_values import (
    drop_high_missing_data,
    fill_missing_categorical,
    fill_missing_numerical,
)
from src.preprocess.text_processing import process_text


def preprocess_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력된 df에 대해 동적 전처리를 수행한다.
    - detect_features()로 컬럼 분류
    - 결측치 처리
    - 이상치 처리
    - 스케일링
    - 인코딩
    - 특성 생성
    :return: 전처리가 완료된 데이터프레임
    """

    # 1. 데이터 특성 탐지
    feature_info = detect_features(df)
    cat_cols = feature_info['categorical']
    num_cols = feature_info['numerical']
    datetime_cols = feature_info['datetime']
    text_cols = feature_info['text']
    geospatial_cols = feature_info['geospatial']
    dtype_info = feature_info['dtypes']
    scaler_info = {col: IdentityScaler() for col in df.columns}


    # 2. 결측치 처리
    df = drop_high_missing_data(df, threshold=0.5)
    df = fill_missing_numerical(df, num_cols, strategy='median')
    df = fill_missing_categorical(df, cat_cols, fill_value='Unknown')

    if text_cols:
        df = fill_missing_categorical(df, text_cols, fill_value='Unknown')

    # 5. 텍스트 데이터 처리
    if text_cols:
        df = process_text(df, text_cols)

    # 8. 인코딩 (동적 처리)
    df, scaler_info = dynamic_encode(df, feature_info, scaler_info)

    # 3. 이상치 처리 (동적 처리)
    df = dynamic_outlier_removal(df, num_cols)

    # 4. 날짜형 데이터 처리
    if datetime_cols:
        df = remove_datetime_columns(df, datetime_cols)

    # 6. 지리정보형 데이터 처리
    if geospatial_cols:
        df = remove_geospatial_columns(df, geospatial_cols)

    # 7. 스케일링 (동적 처리)
    df, scaler_info = dynamic_scaling(df, num_cols, scaler_info)

    # 전처리 완료된 데이터프레임 반환
    return {
        'processed_df': df,
        'dtypes': dtype_info,
        'scalers': scaler_info
    }