# src/preprocess/outlier_detection.py

import fireducks.pandas as pd

def remove_outliers_iqr(df: pd.DataFrame, numerical_cols: list, factor: float = 1.5) -> pd.DataFrame:
    """
    IQR 방식을 사용해 이상치를 제거한다.
    :param df: 입력 데이터프레임
    :param numerical_cols: 이상치 제거할 수치형 열 목록
    :param factor: IQR 범위에 곱하는 계수 (기본 1.5)
    """
    for col in numerical_cols:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        # 이상치 범위 밖의 데이터 제거
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def remove_outliers_zscore(df: pd.DataFrame, numerical_cols: list, threshold: float = 3.0) -> pd.DataFrame:
    """
    Z-score 방식을 사용해 이상치를 제거한다.
    :param df: 입력 데이터프레임
    :param numerical_cols: 이상치 제거할 수치형 열 목록
    :param threshold: Z-score 임계값 (기본 3.0)
    """
    for col in numerical_cols:
        if col not in df.columns:
            continue
        
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0:
            continue  # 분산이 0이면 모든 값이 같으므로 처리 스킵

        z_scores = (df[col] - mean_val) / std_val
        df = df[z_scores.abs() <= threshold]
    return df
