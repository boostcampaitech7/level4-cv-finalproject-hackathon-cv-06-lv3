# main.py

import argparse
import logging
import time

import src.datasets as datasets
import src.search as search
import src.surrogate as surrogate
from src.utils import Setting, measure_time

# from src.surrogate.eval_surrogate_model import eval_surrogate_model

def main(args):
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 랜덤 시드 설정
    Setting.seed_everything(args.seed)
    
    datatype = args.dataset # 데이터셋
    model_name = args.model # 사용할 서로게이트 모델 명
    search_model = args.search_model # 사용할 서치 모델 명명

    if model_name == 'simpleNN':
        raise ValueError("simpleNN is not supported for now")
    if search_model == 'backprob':
        raise ValueError("backprob is not supported for now")
    
    # 데이터 로드 및 분할
    try:
        load_data_func = getattr(datasets, f'{datatype}_data')
    except AttributeError:
        logging.error(f"지원되지 않는 데이터셋 '{datatype}' 입니다.")
        return

    X_train, X_test, y_train, y_test = load_data_func(args.data_path)

    # 모델별 데이터 로드
    try:
        load_data_loader_func = getattr(datasets, f'{model_name}_load_data')
    except AttributeError:
        logging.error(f"지원되지 않는 모델 '{model_name}' 입니다.")
        return

    train_loader, val_loader = load_data_loader_func(X_train, X_test, y_train, y_test)
    
    # 데이터셋 형태 출력
    logging.info(f"X_train.shape: {X_train.shape}")
    logging.info(f"X_test.shape: {X_test.shape}")
    logging.info(f"y_train.shape: {y_train.shape}")
    logging.info(f"y_test.shape: {y_test.shape}")
    
    # 모델 학습
    try:
        train_func = getattr(surrogate, f'{model_name}_train')
        # model = measure_time(train_func, train_loader, val_loader)
        model = train_func(train_loader, val_loader)
    except AttributeError:
        logging.error(f"지원되지 않는 모델 '{model_name}'의 학습 함수입니다.")
        return

    # 예측 수행
    # try:
    predict_func = getattr(surrogate, f'{model_name}_predict')
    y_pred = predict_func(model, X_test)
    # print(y_pred.shape) # (batch_size, 1)
    # except AttributeError:
    #     logging.error(f"지원되지 않는 모델 '{model_name}'의 예측 함수입니다.")
    #     return

    # 모델 평가
    try:
        rmse, mae, r2 = surrogate.eval_surrogate_model(y_train, y_pred, y_test)
        #TODO mutiple Y일 때 수정!!!!!!
        logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    except Exception as e:
        logging.error(f"모델 평가 중 오류 발생: {e}")
        return

    ### 우선 더미 값 들
    ###TODO 모델에 추가적으로 controllable 범위, controllable 변수 명, 환경 변수 값, 아웃풋 변수 명, 최적화 목표, 프로젝트 아이디, 피쳐 별 우선순위 전달 필요
    preprocessed_data_path = './data/concrete_processed.csv' # 전처리 된 데이터 패스스
    # controllable_name = args.c_name # 리스트
    controllable_name = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
    # controllable_range = args.range # 딕셔너리
    controllable_range = {'cement': (102.0, 540.0),
                'slag': (0.0, 359.4),
                'ash': (0.0, 200.1),
                'water': (121.8, 247.0),
                'superplastic': (0.0, 32.2),
                'coarseagg': (801.0, 1145.0),
                'fineagg': (594.0, 992.6),
                'age': (1.0, 365.0)
                } # {col:(minv,maxv) for col, minv, maxv in zip(df.columns.values, df.min().values, df.max().values)}
                # 지금은 그냥 float 형이지만 나중에 받을 떈 np.float64
    # data_env = args.env # 딕셔너리
    data_env = {}
    # output_name = args.out # 리스트
    output_name = ['strength']
    # pj_id = args.project_id # 인트 값 하나
    pj_id = 13
    # importance = args.importance # 딕셔너리
    importance = {'cement': 2,
                'slag': 3,
                'ash': 1,
                'water': 3,
                'superplastic': 2,
                'coarseagg': 2,
                'fineagg': 3,
                'age': 1
                }
    # optimize = args.optimize # 딕셔너리 각 피쳐 별 목표 (최대화, 최소화 등)
    optimize = {'cement': 'maximize',
                'slag': 'minimize',
                'ash': 'maximize',
                'water': 'maximize',
                'superplastic': 'minimize',
                'coarseagg': 'minimize',
                'fineagg': 'minimize',
                'age': 'minimize'
                }
    # target_output = args.target
    target_output = 66

    # 최적화/검색 수행
    # try:
    search_func = getattr(search, f'{search_model}_search')
    start_time = time.time()
    x_opt = search_func(model, predict_func, X_train, X_test, y_test)
    end_time = time.time()
    print(f"search model 소요 시간: {end_time - start_time:.4f}초")

    # except AttributeError:
    #     logging.error(f"지원되지 않는 검색 모델 '{search_model}' 입니다.")
    #     return

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
        choices=['cement', 'melb', 'car', 'ecommerce'], help='사용할 데이터셋을 지정합니다 (기본값: cement)')
    arg('--data_path', '--data_path', '-data_path', type=str, default='./data/concrete_processed.csv',
        help='데이터셋 CSV 파일 경로를 지정합니다')
    arg('--model', '--model', '-model', type=str, default='lightgbm',
        choices=['lightgbm', 'simpleNN', 'tabpfn'], help='사용할 모델을 지정합니다 (기본값: lightgbm)')
    arg('--search_model', '--search_model', '-search_model', type=str, default='ga_deap',
        choices=['backprob', 'ga_deap', 'ga_pygmo', 'ga_adaptive_niching', 'k_means'], help='사용할 검색/최적화 방법을 지정합니다 (기본값: backprob)')
    arg('--seed', '--seed', '-seed', type=int, default=42,
        help='재현성을 위한 랜덤 시드 (기본값: 42)')
    args = parser.parse_args()

    # TODO: omegaconf 적용

    main(args)
