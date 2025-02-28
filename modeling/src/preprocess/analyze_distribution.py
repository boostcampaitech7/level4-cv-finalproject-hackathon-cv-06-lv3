# src/preprocess/analyze_distribution.py

from scipy.stats import shapiro
import fireducks.pandas as pd

def analyze_distribution(df: pd.DataFrame, numerical_cols: list) -> dict:
    """
    수치형 열의 분포를 분석하여 정규 분포 여부를 반환.
    :param df: 입력 데이터프레임
    :param numerical_cols: 수치형 열 목록
    :return: 각 열에 대한 분포 분석 결과 딕셔너리
    """
    results = {}
    for col in numerical_cols:
        if col not in df.columns:
            continue
        
        skewness = df[col].skew()
        p_value = shapiro(df[col].dropna()).pvalue  # Shapiro-Wilk Test
        is_normal = p_value > 0.05  # p-value > 0.05이면 정규 분포로 간주

        results[col] = {
            'skewness': skewness,
            'p_value': p_value,
            'is_normal': is_normal
        }
    return results
