import pickle
import warnings

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", category=FutureWarning)


def tabpfn_multi_train(train_loader, val_loader):
    """
    MultiOutputRegressor를 사용하여 TabPFNRegressor 모델을 학습합니다.

    Args:
        train_loader (tuple): (X_train, y_train) 형태의 학습 데이터
        val_loader (tuple): (X_test, y_test) 형태의 검증 데이터

    Returns:
        MultiOutputRegressor: 학습된 모델
    """
    model = MultiOutputRegressor(TabPFNRegressor(device="cuda"))
    (X_train, y_train), (X_test, y_test) = train_loader, val_loader
    model.fit(X_train, y_train)

    return model


def tabpfn_multi_predict(model, X_test: np.ndarray) -> np.ndarray:
    """
    학습된 모델을 사용하여 입력 데이터에 대한 예측을 수행합니다.

    Args:
        model (MultiOutputRegressor): 학습된 MultiOutputRegressor 모델
        X_test (np.ndarray): 예측할 입력 데이터

    Returns:
        np.ndarray: 예측된 출력 값
    """

    # print(f'X_test :{X_test.shape}')
    y_pred = model.predict(X_test)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, X_test.shape[1])
    # print(f'y pred : {y_pred.shape}')
    return y_pred


def tabpfn_multi_save(model, path):
    """
    모델을 지정된 경로에 피클(pickle) 파일로 저장합니다.

    Args:
        model (MultiOutputRegressor): 저장할 모델 객체
        path (str): 저장할 파일 경로 (확장자 제외)
    """
    with open(path + ".pkl", "wb") as f:
        pickle.dump(model, f)


def tabpfn_multi_load(path):
    """
    지정된 경로에서 피클(pickle) 파일을 불러와 모델을 반환합니다.

    Args:
        path (str): 불러올 파일 경로 (확장자 제외)

    Returns:
        MultiOutputRegressor: 로드된 모델 객체
    """
    with open(path + ".pkl", "rb") as f:
        return pickle.load(f)
