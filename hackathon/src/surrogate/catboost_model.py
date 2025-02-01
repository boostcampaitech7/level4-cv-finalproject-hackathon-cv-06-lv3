import numpy as np
from catboost import CatBoostRegressor


def catboost_train(train_data: tuple, val_data: tuple, params: dict = None):
    """
    CatBoost 회귀 모델을 학습하는 함수.

    Parameters:
        train_data (tuple): 훈련 데이터 (X_train, y_train)
        val_data (tuple): 검증 데이터 (X_test, y_test)
        params (dict, optional): CatBoost 하이퍼파라미터 딕셔너리. 기본값은 None.

    Returns:
        CatBoostRegressor: 학습된 CatBoost 회귀 모델
    """
    X_train, y_train = train_data
    X_test, y_test = val_data

    # CatBoost 회귀 모델 생성
    model = CatBoostRegressor(
        iterations=2000,  # 최대 반복 횟수
        depth=7,  # 트리 깊이
        learning_rate=0.05,  # 학습률
        bagging_temperature=1,  # 배깅(bootstrap) 샘플링 강도 조절
        loss_function="RMSE",  # 손실 함수 (Root Mean Squared Error)
        random_seed=42,  # 재현성을 위한 랜덤 시드 설정
        verbose=100,  # 학습 로그 출력 간격
        early_stopping_rounds=100,  # 조기 종료 설정
    )

    # 모델 학습
    model.fit(X_train, y_train)
    
    return model

def catboost_predict(model, X_test: np.ndarray) -> np.ndarray:
    """
    학습된 CatBoost 회귀 모델을 사용하여 예측 수행.

    Parameters:
        model (CatBoostRegressor): 학습된 CatBoost 회귀 모델
        X_test (np.ndarray): 예측을 수행할 입력 데이터

    Returns:
        np.ndarray: 예측된 출력 값
    """
    y_pred = model.predict(X_test)  # 모델을 사용하여 예측 수행

    # 예측 결과가 1차원 배열이면 2차원으로 변환
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return y_pred

def catboost_save(model, path):
    """
    CatBoost 모델을 지정된 경로에 저장합니다.

    Args:
        model (CatBoostRegressor): 저장할 CatBoost 모델 객체
        path (str): 모델을 저장할 파일 경로

    Returns:
        None
    """
    model.save_model(path, format='cbm')

def catboost_load(path):
    """
    지정된 경로에서 CatBoost 모델을 로드합니다.

    Args:
        path (str): 로드할 모델 파일 경로

    Returns:
        CatBoostRegressor: 로드된 CatBoost 모델 객체
    """
    model = CatBoostRegressor()
    model.load_model(path)
    return model
