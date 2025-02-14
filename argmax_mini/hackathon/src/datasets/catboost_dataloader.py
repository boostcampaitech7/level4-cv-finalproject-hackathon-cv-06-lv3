import numpy as np


def catboost_load_data(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
):
    """
    학습 및 테스트 데이터를 반환하는 함수

    Args:
        X_train (np.ndarray): 학습 데이터의 입력 피처 (형태: (n_samples, n_features))
        X_test (np.ndarray): 테스트 데이터의 입력 피처 (형태: (n_samples, n_features))
        y_train (np.ndarray): 학습 데이터의 타깃 값 (형태: (n_samples,))
        y_test (np.ndarray): 테스트 데이터의 타깃 값 (형태: (n_samples,))

    Returns:
        tuple: ((X_train, y_train), (X_test, y_test)) 형식의 튜플을 반환
    """
    return (X_train, y_train), (X_test, y_test)
