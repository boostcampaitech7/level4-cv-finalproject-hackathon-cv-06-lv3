import fireducks.pandas as pd

def remove_outliers_iqr(df: pd.DataFrame, numerical_cols: list, factor: float = 1.5) -> pd.DataFrame:
    """
    IQR 방식을 사용해 이상치를 제거한다.
    여러 열에 대해 이상치 인덱스를 모아서 한 번에 삭제함으로써 메모리 사용을 최소화한다.
    
    :param df: 입력 데이터프레임 (inplace 수정)
    :param numerical_cols: 이상치 제거할 수치형 열 목록
    :param factor: IQR 범위에 곱하는 계수 (기본 1.5)
    :return: 이상치가 제거된 데이터프레임
    """
    outlier_indices = set()
    for col in numerical_cols:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_indices.update(indices)
        
    # 한 번에 행 삭제
    if outlier_indices:
        df.drop(index=list(outlier_indices), inplace=True)
    return df


def remove_outliers_zscore(df: pd.DataFrame, numerical_cols: list, threshold: float = 3.0) -> pd.DataFrame:
    """
    Z-score 방식을 사용해 이상치를 제거한다.
    여러 열에 대해 이상치 인덱스를 모아서 한 번에 삭제함으로써 메모리 사용을 최소화한다.
    
    :param df: 입력 데이터프레임 (inplace 수정)
    :param numerical_cols: 이상치 제거할 수치형 열 목록
    :param threshold: Z-score 임계값 (기본 3.0)
    :return: 이상치가 제거된 데이터프레임
    """
    outlier_indices = set()
    for col in numerical_cols:
        if col not in df.columns:
            continue

        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0:
            continue  # 모든 값이 동일하면 넘어감

        z_scores = (df[col] - mean_val) / std_val
        indices = df.index[z_scores.abs() > threshold]
        outlier_indices.update(indices)
        
    if outlier_indices:
        df.drop(index=list(outlier_indices), inplace=True)
    return df
