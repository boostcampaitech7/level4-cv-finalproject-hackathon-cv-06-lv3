# main.py

import argparse
import logging

from src.utils import Setting, measure_time
import src.datasets as datasets
import src.surrogate as surrogate
import src.search as search
# from src.surrogate.eval_surrogate_model import eval_surrogate_model

def main(args):
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 랜덤 시드 설정
    Setting.seed_everything(args.seed)
    
    datatype = args.dataset
    model_name = args.model
    search_model = args.search_model

    # 데이터 로드 및 분할
    try:
        load_data_func = getattr(datasets, f'{datatype}_data')
    except AttributeError:
        logging.error(f"지원되지 않는 데이터셋 '{datatype}' 입니다.")
        return

    X_train, X_test, y_train, y_test = load_data_func(args.data_path)

    # 모델별 데이터 로드
    try:
        load_model_data_func = getattr(datasets, f'{model_name}_load_data')
    except AttributeError:
        logging.error(f"지원되지 않는 모델 '{model_name}' 입니다.")
        return

    train_data, val_data = load_model_data_func(X_train, X_test, y_train, y_test)
    
    # 데이터셋 형태 출력
    logging.info(f"X_train.shape: {X_train.shape}")
    logging.info(f"X_test.shape: {X_test.shape}")
    logging.info(f"y_train.shape: {y_train.shape}")
    logging.info(f"y_test.shape: {y_test.shape}")
    
    # 모델 학습
    try:
        train_func = getattr(surrogate, f'{model_name}_train')
        model = measure_time(train_func, train_data, val_data)
    except AttributeError:
        logging.error(f"지원되지 않는 모델 '{model_name}'의 학습 함수입니다.")
        return

    # 예측 수행
    try:
        predict_func = getattr(surrogate, f'{model_name}_predict')
        y_pred = predict_func(model, val_data)
    except AttributeError:
        logging.error(f"지원되지 않는 모델 '{model_name}'의 예측 함수입니다.")
        return

    # 모델 평가
    try:
        rmse, mae, r2 = surrogate.eval_surrogate_model(y_train, y_pred, y_test)
        logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    except Exception as e:
        logging.error(f"모델 평가 중 오류 발생: {e}")
        return

    # 최적화/검색 수행
    try:
        search_func = getattr(search, f'{search_model}_search')
        x_opt = search_func(model, predict_func, X_train, val_data)
    except AttributeError:
        logging.error(f"지원되지 않는 검색 모델 '{search_model}' 입니다.")
        return

    # 최적화 결과 평가
    try:
        rmse, mae, r2 = search.eval_search_model(X_train, x_opt, X_test)
        logging.info("R²: " + ", ".join([f"{x:.2f}" for x in r2]))
    except Exception as e:
        logging.error(f"최적화 결과 평가 중 오류 발생: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 학습 및 최적화 스크립트')
    arg = parser.add_argument
    arg('--dataset', '--dset', '-dset', type=str, default='cement',
        choices=['cement', 'melb'], help='사용할 데이터셋을 지정합니다 (기본값: cement)')
    arg('--data_path', '--data_path', '-data_path', type=str, default='./data/concrete_processed.csv',
        help='데이터셋 CSV 파일 경로를 지정합니다')
    arg('--model', '--model', '-model', type=str, default='lightgbm',
        choices=['lightgbm','simpleNN'], help='사용할 모델을 지정합니다 (기본값: lightgbm)')
    arg('--search_model', '--search_model', '-search_model', type=str, default='backprob',
        choices=['backprob'], help='사용할 검색/최적화 방법을 지정합니다 (기본값: backprob)')
    arg('--seed', '--seed', '-seed', type=int, default=42,
        help='재현성을 위한 랜덤 시드 (기본값: 42)')
    args = parser.parse_args()

    # TODO: omegaconf 적용

    main(args)
