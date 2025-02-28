# src/preprocess/dynamic_outlier.py

from src.preprocess.outlier_detection import remove_outliers_iqr, remove_outliers_zscore
from src.preprocess.analyze_distribution import analyze_distribution
import fireducks.pandas as pd

def dynamic_outlier_removal(df: pd.DataFrame, numerical_cols: list, **kwargs) -> pd.DataFrame:
    """
    분포 분석을 기반으로 동적으로 이상치 처리 전략을 선택한다.
    :param df: 입력 데이터프레임
    :param numerical_cols: 수치형 열 목록
    :param kwargs: 이상치 처리 파라미터
        - zscore_threshold: Z-score 방식 임계값 (기본값: 3.0)
        - iqr_factor: IQR 방식 계수 (기본값: 1.5)
    :return: 이상치가 처리된 데이터프레임
    """
    # 분포 분석
    distribution_info = analyze_distribution(df, numerical_cols)

    for col in numerical_cols:
        if col not in df.columns:
            continue
        
        info = distribution_info[col]
        if info['is_normal']:
            # 정규 분포: Z-score 방식
            print(f"{col}: 정규 분포로 확인됨. Z-score 방식 적용.")
            df = remove_outliers_zscore(df, [col], threshold=kwargs.get('zscore_threshold', 3.0))
        else:
            # 비대칭 분포: IQR 방식
            print(f"{col}: 비대칭 분포로 확인됨. IQR 방식 적용.")
            df = remove_outliers_iqr(df, [col], factor=kwargs.get('iqr_factor', 1.5))
    
    return df
