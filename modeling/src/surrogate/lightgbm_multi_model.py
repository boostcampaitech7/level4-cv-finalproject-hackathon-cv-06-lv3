import joblib
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor


def lightgbm_multi_train(train_data: tuple, val_data: tuple, params: dict = None):
    """
    LightGBM 다중 출력 회귀 모델을 학습하는 함수

    Parameters:
        train_data (tuple): 훈련 데이터 (X_train, y_train)
        val_data (tuple): 검증 데이터 (X_test, y_test)
        params (dict, optional): LightGBM 하이퍼파라미터 딕셔너리. 기본값은 None.

    Returns:
        MultiOutputRegressor: 학습된 다중 출력 회귀 모델
    """
    X_train, y_train = train_data
    X_test, y_test = val_data
    
    # 기본 하이퍼파라미터 설정
    if params is None:
        params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
        }
    
    # LightGBM 기본 회귀 모델 설정
    base_estimator = LGBMRegressor(**params, n_estimators=1000)
    
    # 다중 출력 회귀 모델 생성
    multi_model = MultiOutputRegressor(base_estimator)
    
    # 모델 학습
    multi_model.fit(X_train, y_train)
    
    return multi_model

def lightgbm_multi_predict(model, X_test: np.ndarray) -> np.ndarray:
    """
    학습된 LightGBM 다중 출력 회귀 모델을 사용하여 예측 수행

    Parameters:
        model (MultiOutputRegressor): 학습된 다중 출력 회귀 모델
        X_test (np.ndarray): 예측을 수행할 입력 데이터

    Returns:
        np.ndarray: 예측된 출력 값
    """
    return model.predict(X_test)

def lightgbm_multi_save(model, path):
    """
    모델 객체를 지정된 경로(.pkl 확장자 포함)로 저장합니다.

    :param model: 저장할 모델 객체
    :param path: 모델을 저장할 파일 경로 (확장자 제외)
    """
    joblib.dump(model, path + '.pkl')


def lightgbm_multi_load(path):
    """
    지정된 경로(.pkl 확장자 포함)에서 모델 객체를 로드합니다.

    :param path: 모델을 로드할 파일 경로 (확장자 제외)
    :return: 로드된 모델 객체
    """
    return joblib.load(path + '.pkl')