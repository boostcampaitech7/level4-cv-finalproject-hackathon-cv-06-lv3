# src/preprocess/dynamic_outlier.py

from hackathon.src.preprocess.outlier_detection import remove_outliers_iqr, remove_outliers_zscore
from hackathon.src.preprocess.analyze_distribution import analyze_distribution
import fireducks.pandas as pd


def dynamic_outlier_removal(df: pd.DataFrame, numerical_cols: list, **kwargs) -> pd.DataFrame:
    """
    분포 분석을 기반으로 동적으로 이상치 처리 전략을 선택한다.
    메모리 사용을 최적화하기 위해 각 방식별로 여러 열에 대해
    이상치 인덱스를 모아서 한 번에 삭제한다.

    :param df: 입력 데이터프레임
    :param numerical_cols: 수치형 열 목록
    :param kwargs: 이상치 처리 파라미터
        - zscore_threshold: Z-score 방식 임계값 (기본값: 3.0)
        - iqr_factor: IQR 방식 계수 (기본값: 1.5)
    :return: 이상치가 처리된 데이터프레임
    """
    # 분포 분석
    distribution_info = analyze_distribution(df, numerical_cols)

    # Z-score 및 IQR로 처리할 열 구분
    zscore_cols = [col for col in numerical_cols if col in df.columns and distribution_info[col]['is_normal']]
    iqr_cols = [col for col in numerical_cols if col in df.columns and not distribution_info[col]['is_normal']]

    # Z-score 적용 (정규 분포)
    if zscore_cols:
        print(f"정규 분포 확인된 열: {zscore_cols}, Z-score 방식 적용.")
        remove_outliers_zscore(df, zscore_cols, threshold=kwargs.get('zscore_threshold', 3.0))

    # IQR 적용 (비대칭 분포)
    if iqr_cols:
        print(f"비대칭 분포 확인된 열: {iqr_cols}, IQR 방식 적용.")
        remove_outliers_iqr(df, iqr_cols, factor=kwargs.get('iqr_factor', 1.5))

    return df


