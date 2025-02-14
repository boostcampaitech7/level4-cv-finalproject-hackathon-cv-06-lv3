

import numpy as np
from .data_loader import load_data, split_data  # 프로젝트 구조에 맞게 임포트 수정

def load_and_split_data(file_path: str, target: str):
    """
    데이터셋을 로드하고 훈련 및 테스트 세트로 분할합니다.

    Args:
        file_path (str): 전처리된 CSV 파일 경로.
        target (str): 타겟 컬럼 이름.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, X_test, y_train, y_test
    """
    # 데이터 로드
    df = load_data(file_path)

    # 데이터 분할
    X_train, X_test, y_train, y_test = split_data(df, target=target)

    # 타겟을 넘파이 배열로 변환
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)


    return X_train, X_test, y_train, y_test

def load_and_split_data_with_x_col_list(file_path: str, target: list):
    """
    데이터셋을 로드하고 훈련 및 테스트 세트로 분할합니다.

    Args:
        file_path (str): 전처리된 CSV 파일 경로.
        target (str): 타겟 컬럼 이름.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, X_test, y_train, y_test
    """
    # 데이터 로드
    df = load_data(file_path)

    # 데이터 분할
    X_train, X_test, y_train, y_test = split_data(df, target=target)

    # 타겟을 넘파이 배열로 변환
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)


    return X_train, X_test, y_train, y_test, df.drop(columns=target).columns.tolist()

def cement_data(file_path: str):
    """
    시멘트 데이터셋을 로드하고 분할합니다.

    Args:
        file_path (str): 시멘트 CSV 파일 경로.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    return load_and_split_data(file_path, target="strength")

def melb_data(file_path: str):
    """
    멜버른 데이터셋을 로드하고 분할합니다.

    Args:
        file_path (str): 멜버른 CSV 파일 경로.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    return load_and_split_data(file_path, target="Price")

def car_data(file_path: str):
    """
    멜버른 데이터셋을 로드하고 분할합니다.

    Args:
        file_path (str): 멜버른 CSV 파일 경로.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    return load_and_split_data(file_path, target="Annual_revenue")

def ecommerce_data(file_path: str):
    """
    Ecommerce 데이터셋을 로드하고 분할합니다.

    Args:
        file_path (str): Ecommerce Parquet 파일 경로.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    return load_and_split_data(file_path, target="revenue")
