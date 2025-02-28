# src/preprocess/missing_values.py

import fireducks.pandas as pd
from hackathon.src.preprocess.processing_metadata import add_removed_columns


def fill_missing_numerical(df: pd.DataFrame, numerical_cols: list, strategy: str = 'median') -> pd.DataFrame:
    """
    수치형 열에 대해 결측치를 특정 전략(평균, 중앙값 등)으로 채운다. (메모리 최적화 적용)
    
    :param df: 입력 데이터프레임
    :param numerical_cols: 수치형 열 목록
    :param strategy: 'mean' or 'median'
    """
    # 1. 데이터프레임에 해당 컬럼들이 존재하는지 확인 후 필터링
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if not numerical_cols:
        return df  # 처리할 컬럼이 없으면 그대로 반환

    # 2. mean/median 연산을 한 번만 수행하여 메모리 최적화
    if strategy == 'mean':
        fill_values = df[numerical_cols].mean()
    elif strategy == 'median':
        fill_values = df[numerical_cols].median()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 3. DataFrame 전체에 fillna() 한 번만 실행 (메모리 절약)
    df[numerical_cols] = df[numerical_cols].fillna(fill_values)

    return df


def fill_missing_categorical(df: pd.DataFrame, categorical_cols: list, fill_value: str = 'Unknown') -> pd.DataFrame:
    """
    범주형 열에 대해 결측치를 특정 값으로 채운다.
    메모리 초과를 방지하기 위해 inplace=True로 결측치를 채운다.
    :param df: 입력 데이터프레임
    :param categorical_cols: 범주형 열 목록
    :param fill_value: 결측치 대체값 (기본값 'Unknown')
    """
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(fill_value, inplace=True)
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
        df.drop(columns=drop_cols, inplace=True)
    else:
        print("제거된 컬럼이 없습니다.")

    return df
