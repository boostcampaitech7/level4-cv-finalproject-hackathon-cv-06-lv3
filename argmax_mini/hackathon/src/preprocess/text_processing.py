# src/preprocess/text_processing.py

import fireducks.pandas as pd
import re

def process_text(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    텍스트 컬럼에 대한 전처리: 소문자 변환, 특수문자 제거 등.
    메모리 사용량을 줄이고 성능을 개선하기 위해 vectorized string 연산을 체인 형태로 사용합니다.
    
    :param df: 입력 데이터프레임
    :param text_columns: 전처리할 텍스트 컬럼 목록
    :return: 전처리된 텍스트 컬럼을 포함한 데이터프레임
    """
    for col in text_columns:
        if col not in df.columns:
            continue
        # 문자열로 변환, 소문자화, 정규표현식을 이용한 특수문자 제거를 하나의 체인으로 수행합니다.
        df[col] = df[col].astype(str).str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    return df
