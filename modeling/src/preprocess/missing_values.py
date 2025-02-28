# src/preprocess/missing_values.py

import fireducks.pandas as pd
from src.preprocess.processing_metadata import add_removed_columns

def fill_missing_numerical(df: pd.DataFrame, numerical_cols: list, strategy: str = 'median') -> pd.DataFrame:
    """
    수치형 열에 대해 결측치를 특정 전략(평균, 중앙값 등)으로 채운다.
    :param df: 입력 데이터프레임
    :param numerical_cols: 수치형 열 목록
    :param strategy: 'mean' or 'median'
    """
    for col in numerical_cols:
        if col not in df.columns:
            continue

        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # loc를 사용하여 체인 할당 방지
        df.loc[:, col] = df[col].fillna(fill_value)
    return df

def fill_missing_categorical(df: pd.DataFrame, categorical_cols: list, fill_value: str = 'Unknown') -> pd.DataFrame:
    """
    범주형 열에 대해 결측치를 특정 값으로 채운다.
    :param df: 입력 데이터프레임
    :param categorical_cols: 범주형 열 목록
    :param fill_value: 결측치 대체값 (기본값 'Unknown')
    """
    for col in categorical_cols:
        if col not in df.columns:
            continue
        # loc를 사용하여 체인 할당 방지
        df.loc[:, col] = df[col].fillna(fill_value)
    return df

def drop_high_missing_data(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    결측치 비율이 threshold를 초과하는 컬럼을 제거한다.
    :param df: 입력 데이터프레임
    :param threshold: 허용되는 결측치 비율 (0.5 = 50%)
    :return: 결측치 비율 기준으로 정리된 데이터프레임
    """
    # 컬럼 단위로 결측치 비율 계산
    missing_col_ratio = df.isnull().mean()
    drop_cols = missing_col_ratio[missing_col_ratio > threshold].index.tolist()
    
    # 제거된 컬럼 기록
    if drop_cols:
        add_removed_columns(drop_cols)
        print(f"제거된 컬럼: {drop_cols}")
        df = df.drop(columns=drop_cols)
    else:
        print("제거된 컬럼이 없습니다.")

    return df