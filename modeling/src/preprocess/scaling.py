# src/preprocess/scaling.py

import fireducks.pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_data(df: pd.DataFrame, numerical_cols: list, method: str = 'standard'):
    """
    수치형 데이터를 스케일링한다.
    :param df: 입력 데이터프레임
    :param numerical_cols: 스케일링할 수치형 열 목록
    :param method: 'standard', 'minmax', 'robust'
    :return: (스케일링된 df, scaler) 튜플
    """
    if not numerical_cols:
        return df, None  # 스케일링할 열이 없으면 그대로 반환

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
