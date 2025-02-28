# src/preprocess/dynamic_scaling.py

import fireducks.pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from src.preprocess.analyze_distribution import analyze_distribution


def dynamic_scaling(df: pd.DataFrame, numerical_cols: list, scalers: dict) -> pd.DataFrame:
    """
    데이터 특성에 따라 동적으로 스케일링 방법을 선택.
    :param df: 입력 데이터프레임
    :param numerical_cols: 스케일링할 수치형 열 목록
    :return: 스케일링된 데이터프레임
    """
    distribution_info = analyze_distribution(df, numerical_cols)
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
  
        info = distribution_info[col]
        if info['is_normal']:
            print(f"{col}: 정규분포 → StandardScaler 적용")
            scaler = StandardScaler()
        elif abs(info['skewness']) > 1.0:
            print(f"{col}: 비대칭 분포 → RobustScaler 적용")
            scaler = RobustScaler()
        else:
            print(f"{col}: 값 범위 중요 → MinMaxScaler 적용")
            scaler = MinMaxScaler()
        
        # 해당 열 스케일링 및 스케일러 저장
        df[[col]] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler  # 나중에 동일한 스케일링 적용 가능
    
    return df, scalers
