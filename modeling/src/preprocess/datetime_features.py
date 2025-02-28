# src/preprocess/datetime_features.py

import fireducks.pandas as pd
from src.preprocess.processing_metadata import add_removed_columns

from src.preprocess.processing_metadata import add_removed_columns

def remove_datetime_columns(df: pd.DataFrame, datetime_columns: list) -> pd.DataFrame:
    """
    날짜형 컬럼 제거 및 제거된 컬럼 기록
    :param df: 입력 데이터프레임
    :param datetime_columns: 제거할 날짜형 컬럼 목록
    :return: 날짜형 컬럼이 제거된 데이터프레임
    """
    # 제거할 날짜형 컬럼이 있는지 확인
    columns_to_remove = [col for col in datetime_columns if col in df.columns]
    if columns_to_remove:
        # 제거된 컬럼 기록
        add_removed_columns(columns_to_remove)
        print(f"제거된 날짜형 컬럼: {columns_to_remove}")
        # 데이터프레임에서 컬럼 제거
        df = df.drop(columns=columns_to_remove)
    else:
        print("제거된 날짜형 컬럼이 없습니다.")

    return df

