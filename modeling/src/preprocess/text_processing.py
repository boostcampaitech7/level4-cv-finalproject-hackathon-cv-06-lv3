# src/preprocess/text_processing.py

import fireducks.pandas as pd
import re

def process_text(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    텍스트 컬럼에 대한 전처리: 소문자 변환, 특수문자 제거 등
    :param df: 입력 데이터프레임
    :param text_columns: 텍스트 컬럼 목록
    :return: 전처리된 텍스트 컬럼을 포함한 데이터프레임
    """
    for col in text_columns:
        # 텍스트 데이터가 문자열인지 확인
        if not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
        
        # 소문자 변환
        df[col] = df[col].str.lower()
        
        # 특수문자 제거
        df[col] = df[col].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
        
        # 추가 전처리: 불용어 제거, 토큰화 등 필요 시 구현
    
    return df
