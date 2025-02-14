import numpy as np

import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


def get_objective(X_train, y_train, X_test, y_test):
    """
    Optuna 최적화에 사용될 목적 함수를 반환하는 함수

    Args:
        X_train: 훈련 데이터의 특성 행렬
        y_train: 훈련 데이터의 타겟 값
        X_test: 테스트 데이터의 특성 행렬
        y_test: 테스트 데이터의 타겟 값

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
            "objective": trial.suggest_categorical("objective", ["RMSE"]),
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


def catboost_train(train_data: tuple, val_data: tuple, params: dict = None):
    """
    CatBoost 회귀 모델을 학습하는 함수

    Args:
        train_data (tuple): 훈련 데이터 (X_train, y_train)
        val_data (tuple): 검증 데이터 (X_test, y_test)
        params (dict, optional): CatBoost 하이퍼파라미터 딕셔너리. 기본값은 None

    Returns:
        CatBoostRegressor: 학습된 CatBoost 회귀 모델
    """
    X_train, y_train = train_data
    X_test, y_test = val_data

    objective = get_objective(X_train, y_train, X_test, y_test)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    # CatBoost 회귀 모델 생성
    # model = CatBoostRegressor(
    #     iterations=2000,  # 최대 반복 횟수
    #     depth=7,  # 트리 깊이
    #     learning_rate=0.05,  # 학습률
    #     bagging_temperature=1,  # 배깅(bootstrap) 샘플링 강도 조절
    #     loss_function="RMSE",  # 손실 함수 (Root Mean Squared Error)
    #     random_seed=42,  # 재현성을 위한 랜덤 시드 설정
    #     verbose=100,  # 학습 로그 출력 간격
    #     early_stopping_rounds=100,  # 조기 종료 설정
    # )
    model = CatBoostRegressor(**study.best_params)
    # 모델 학습
    model.fit(X_train, y_train)

    return model


def catboost_predict(model, X_test: np.ndarray) -> np.ndarray:
    """
    학습된 CatBoost 회귀 모델을 사용하여 예측 수행

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
    model.save_model(path + ".cbm", format="cbm")


def catboost_load(path):
    """
    지정된 경로에서 CatBoost 모델을 로드합니다.

    Args:
        path (str): 로드할 모델 파일 경로

    Returns:
        CatBoostRegressor: 로드된 CatBoost 모델 객체
    """
    model = CatBoostRegressor()
    model.load_model(path + ".cbm")
    return model
