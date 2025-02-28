import numpy as np

import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


def get_objective(X_train, y_train, X_test, y_test):
    """
    Optuna 최적화에 사용될 목적 함수를 반환하는 함수

    Args:
        X_train: 훈련 데이터 행렬
        y_train: 훈련 데이터 타겟 값
        X_test: 테스트 데이터 행렬
        y_test: 테스트 데이터 타겟 값

    Returns:
        objective (function): Optuna의 목적 함수
    """

    def objective(trial):
        """
        Optuna의 trial 객체를 사용하여 하이퍼파라미터를 최적화하는 함수

        Args:
            trial: Optuna가 제공하는 trial 객체

        Returns:
            float: (1 - 정확도) 값, Optuna는 기본적으로 최소화를 목표로 하므로 1에서 정확도를 뺀 값을 반환
        """
        # data, target = load_breast_cancer(return_X_y=True)
        # train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.3)

        param = {
            "objective": trial.suggest_categorical("objective", ["MultiRMSE"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
        }

        gbm = CatBoostRegressor(**param)

        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=0,
            early_stopping_rounds=100,
        )

        preds = gbm.predict(X_test)
        accuracy = mean_squared_error(y_test, preds)
        return accuracy

    return objective


def catboost_multi_train(train_data: tuple, val_data: tuple, params: dict = None):
    """
    CatBoostRegressor를 사용하여 다중 출력 회귀 모델을 학습하는 함수

    Args:
        train_data (tuple): 훈련 데이터 (X_train, y_train)
        val_data (tuple): 검증 데이터 (X_test, y_test)
        params (dict, optional): CatBoostRegressor의 하이퍼파라미터. 기본값은 None.

    Returns:
        CatBoostRegressor: 학습된 CatBoost 모델
    """
    X_train, y_train = train_data
    X_test, y_test = val_data

    objective = get_objective(X_train, y_train, X_test, y_test)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
    print(study.best_params)
    print(study.best_value)

    model = CatBoostRegressor(**study.best_params)
    #     loss_function="MultiRMSE",  # 다중 출력 회귀를 위한 손실 함수
    #     random_seed=42,  # 랜덤 시드 설정 (재현성 확보)
    #     verbose=200,  # 학습 진행 상황 출력 간격
    # )
    model.fit(X_train, y_train)  # 모델 학습 수행

    return model


def catboost_multi_predict(model, X_test: np.ndarray) -> np.ndarray:
    """
    학습된 CatBoost 모델을 사용하여 예측을 수행하는 함수

    Args:
        model (CatBoostRegressor): 학습된 CatBoost 모델
        X_test (np.ndarray): 예측할 입력 데이터

    Returns:
        np.ndarray: 예측된 다중 출력 회귀 결과
    """
    y_pred = model.predict(X_test) 
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, X_test.shape[1])

    return y_pred # 모델을 사용하여 예측 수행


def catboost_multi_save(model, path):
    """
    CatBoost 모델을 지정된 경로에 저장합니다.

    Args:
        model (CatBoostRegressor): 저장할 CatBoost 모델 객체
        path (str): 모델을 저장할 파일 경로

    Returns:
        None
    """
    model.save_model(path, format="cbm")


def catboost_multi_load(path):
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
