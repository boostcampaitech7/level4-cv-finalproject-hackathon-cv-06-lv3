# optimization/bayesian_optimize.py

import os
import numpy as np
import fireducks.pandas as pd
import lightgbm as lgb
from joblib import load
from bayes_opt import BayesianOptimization
import time  # 시간 측정을 위한 모듈 추가

# 모델과 스케일러의 경로 설정
MODEL_PATH = "/data/ephemeral/home/lightgbm_model.txt"  # 실제 모델 파일 경로로 수정
SCALER_PATH = "/data/ephemeral/home/scaler.pkl"         # 실제 스케일러 파일 경로로 수정

# 학습된 LightGBM 모델 불러오기
def load_model(model_path):
    model = lgb.Booster(model_file=model_path)
    return model

# 스케일러 불러오기
def load_scaler(scaler_path):
    scaler = load(scaler_path)
    return scaler

# 베이지안 최적화를 위한 목적 함수 정의
def objective(cement, water, model, scaler, fixed_params, target_strength=50):
    """
    목적 함수: 예측된 strength와 목표 strength 간의 절대 차이를 최소화
    
    매개변수:
    - cement (float): 시멘트 양
    - water (float): 물의 양
    - model (lgb.Booster): 학습된 LightGBM 모델
    - scaler (sklearn 스케일러): 특성 스케일링을 위한 스케일러
    - fixed_params (dict): 고정된 다른 제어 변수들
    - target_strength (float): 목표 strength 값
    
    반환값:
    - float: 음수 절대 차이 (베이지안 최적화는 최대화를 수행하기 때문에 음수 반환)
    """
    try:
        # 입력 데이터 준비
        input_data = fixed_params.copy()
        input_data['cement'] = cement
        input_data['water'] = water
        
        # 피처의 순서와 이름을 맞추기 위해 리스트로 정의
        feature_names = ["cement", "slag", "ash", "water", "superplastic", "coarseagg", "fineagg", "age"]
        
        # DataFrame 생성 및 피처 순서 맞추기
        df = pd.DataFrame([input_data])[feature_names]
        
        # 특성 스케일링
        scaled_features = scaler.transform(df)
        
        # strength 예측
        y_pred = model.predict(scaled_features)[0]
        
        # 목표 strength와의 차이 계산
        difference = abs(y_pred - target_strength)
        
        # 베이지안 최적화는 최대화를 수행하므로 음수 반환
        return -difference
    except Exception as e:
        print(f"Error in objective function: {e}")
        return -1e6  # 매우 작은 값 반환하여 최적화 과정에서 무시되도록 함

def main():
    # 모델과 스케일러 불러오기
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    
    # 고정된 제어 변수들 정의 (필요에 따라 수정 가능)
    fixed_params = {
        "slag": 100,           # 실제 데이터에 맞게 수정
        "ash": 100,            # 실제 데이터에 맞게 수정
        "superplastic": 10,    # 실제 데이터에 맞게 수정
        "coarseagg": 900,     # 실제 데이터에 맞게 수정
        "fineagg": 600,        # 실제 데이터에 맞게 수정
        "age": 28,             # 실제 데이터에 맞게 수정
        # 'strength'는 목표 변수로 입력 데이터에 포함되지 않음
    }
    
    # 최적화할 변수들의 범위 정의
    pbounds = {
        'cement': (102, 540),  # 데이터에 맞게 범위 조정
        'water': (121, 237)    # 데이터에 맞게 범위 조정
    }
    
    # 베이지안 최적화 초기화
    optimizer = BayesianOptimization(
        f=lambda cement, water: objective(cement, water, model, scaler, fixed_params),
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    # 최적화 시작 시간 기록
    start_time = time.time()
    print("베이지안 최적화 시작...")
    
    # 최적화 수행
    optimizer.maximize(
        init_points=10,  # 초기 랜덤 포인트 수
        n_iter=30,       # 최적화 반복 횟수
    )
    
    # 최적화 종료 시간 기록
    end_time = time.time()
    elapsed_time = end_time - start_time  # 소요 시간 계산
    
    # 최적 결과 추출
    best_result = optimizer.max
    best_cement = best_result['params']['cement']
    best_water = best_result['params']['water']
    predicted_strength = 50 + best_result['target']  # 목표가 음수로 반환되었으므로 더함
    
    print("\n목표 Strength 50을 달성하기 위한 최적의 제어 변수:")
    print(f"Cement: {best_cement:.2f}")
    print(f"Water: {best_water:.2f}")
    print(f"예측된 Strength: {predicted_strength:.2f}")
    
    # 최적화 소요 시간 출력
    print(f"\n최적화 소요 시간: {elapsed_time:.2f}초")

if __name__ == "__main__":
    main()
