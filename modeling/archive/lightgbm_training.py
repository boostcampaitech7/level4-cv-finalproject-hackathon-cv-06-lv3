import lightgbm as lgb
import fireducks.pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
import time
import os

# 1. 데이터 전처리

# 2. LightGBM 학습
def train_lightgbm(X_train, X_test, y_train, y_test):
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 모델 파라미터 설정
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

    # 학습 시간 측정
    print("Training the LightGBM model...")
    start_time = time.time()
    
    try:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,                # 최대 부스팅 반복 수
            valid_sets=[train_data, val_data],   # 학습 및 검증 데이터
            valid_names=["train", "valid"],      # 이름 지정
            callbacks=[
                lgb.log_evaluation(period=100),  # 100번마다 로그 출력
                lgb.early_stopping(stopping_rounds=50)  # 조기 종료 설정
            ]
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

    # 모델 평가
    if model is not None and model.best_iteration > 0:
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation R² Score: {r2:.4f}")
    else:
        print("Model training was unsuccessful or no valid iterations were found.")

    return model

# 3. 모델 저장
def save_model(model, scaler, model_path="lightgbm_model.txt", scaler_path="scaler.pkl"):
    if model is not None and model.best_iteration > 0:
        # 모델 저장
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Model is not available to save.")
    
    # Scaler 저장
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

# Main 함수
def main():
    # 데이터 파일 경로
    file_path = "/data/ephemeral/home/hackerton/data/concrete.csv"

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"Data file not found at {file_path}. Please check the path.")
        return

    # 데이터 전처리
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)

    # LightGBM 학습
    model = train_lightgbm(X_train, X_test, y_train, y_test)

    # 모델 저장
    save_model(model, scaler)

if __name__ == "__main__":
    main()
