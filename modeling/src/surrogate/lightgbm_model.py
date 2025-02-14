import lightgbm as lgb
import numpy as np

def lightgbm_train(train_data, val_data, params = None):

    if params is None:
        params = {
            "objective": "regression",        # 회귀 문제
            "metric": "rmse",                 # 평가 지표
            "boosting_type": "gbdt",          # 부스팅 방식
            "learning_rate": 0.05,            # 학습 속도
            "num_leaves": 31,                 # 리프 노드 개수
            "feature_fraction": 0.8,          # 피처 샘플링 비율
            "bagging_fraction": 0.8,          # 데이터 샘플링 비율
            "bagging_freq": 5,                # 샘플링 빈도
            "min_data_in_leaf": 20,           # 리프 노드 최소 데이터 수
            "verbosity": -1                   # 출력 최소화
        }


    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.log_evaluation(period=100),  # 100번마다 로그 출력
            lgb.early_stopping(stopping_rounds=50)  # 조기 종료 설정
        ]
    )

    return model

def lightgbm_evaluate(model, train_data, val_data):
    # import pdb; pdb.set_trace()
    # X_train = train_data.get_data()
    # print(train_data)
    y_train = train_data.get_label()
    X_test = val_data.get_data()
    y_test = val_data.get_label()

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    
    mae = np.mean(np.abs(y_test - y_pred))
    SSE = np.sum(np.square(y_test - y_pred))    
    SST = np.sum(np.square(y_test - y_train.mean()))
    r2 = 1 - SSE/SST

    rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
    return rmse, mae, r2



def lightgbm_predict(model, X_test: np.ndarray) -> np.ndarray: 
    # X_test shape : (batch_size, variable_num)
    # model : lightgbm.Booster
    # return : (batch_size, 1)
    
    # X_test = X_test.get_data()
    # Tensor를 numpy 배열로 변환
    # X_test = X_test.detach().cpu().numpy()

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return y_pred

def lightgbm_save(model, path):
    
    model.save_model(path+'.txt')

def lightgbm_load(path):
    return lgb.Booster(model_file=path+'.txt')