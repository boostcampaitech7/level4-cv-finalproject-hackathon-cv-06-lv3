import pandas as pd
from hackathon.src.preprocess.sampling import sample_dataframe
from hackathon.src.preprocess.datetime_features import remove_datetime_columns
from hackathon.src.preprocess.detect_features import detect_features
from hackathon.src.preprocess.dynamic_encoding import dynamic_encode
from hackathon.src.preprocess.dynamic_outlier import dynamic_outlier_removal
from hackathon.src.preprocess.dynamic_scaling import dynamic_scaling
# from hackathon.src.preprocess.geospatial_features import remove_geospatial_columns
from hackathon.src.preprocess.identity_scaler import IdentityScaler
from hackathon.src.preprocess.missing_values import drop_high_missing_data, fill_missing_categorical, fill_missing_numerical
from hackathon.src.preprocess.text_processing import process_text


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

    # 1. 샘플링된 데이터 생성 (컬럼 타입 분류용)
    sampled_df = sample_dataframe(df)  # 샘플링된 데이터 사용

    # 2. 데이터 특성 탐지
    feature_info = detect_features(sampled_df)
    cat_cols = feature_info['categorical']
    num_cols = feature_info['numerical']
    num_cat_cols = feature_info['numerical_categorical']
    datetime_cols = feature_info['datetime']
    text_cols = feature_info['text']
    dtype_info = feature_info['dtypes']
    scaler_info = {col: IdentityScaler() for col in df.columns}

    # 백엔드에서 사용할 변수타입별 열 정보 (categorical: 기존 categorical + numerical_categorical)
    combined_cat_cols = cat_cols + num_cat_cols

    # 3. 결측치 처리
    df = drop_high_missing_data(df, threshold=0.5)
    df = fill_missing_numerical(df, num_cols, strategy='median')
    df = fill_missing_categorical(df, cat_cols, fill_value='Unknown')
    
    if num_cat_cols:
        df = fill_missing_numerical(df, num_cat_cols, strategy='median')

    if text_cols:
        df = fill_missing_categorical(df, text_cols, fill_value='Unknown')

    # 4. 텍스트 데이터 처리
    if text_cols:
        df = process_text(df, text_cols)

    # 5. 날짜형 데이터 처리
    if datetime_cols:
        df = remove_datetime_columns(df, datetime_cols)

    # 6. 인코딩 (동적 처리)
    df, scaler_info = dynamic_encode(df, feature_info, scaler_info)

    # 7. 이상치 처리 (동적 처리)
    df = dynamic_outlier_removal(df, num_cols)

    # 8. 스케일링 (동적 처리)
    df_scaled, scaler_info = dynamic_scaling(df, num_cols, scaler_info)

    # 전처리 완료된 데이터프레임 반환
    return df, df_scaled, dtype_info, scaler_info