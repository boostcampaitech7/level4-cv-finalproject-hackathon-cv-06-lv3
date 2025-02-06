# main.py

import argparse
import logging
import time

import src.datasets as datasets
import src.search as search
import src.surrogate as surrogate
from src.utils import Setting, measure_time
from src.datasets.data_loader import load_data

import numpy as np
import pandas as pd
# from src.surrogate.eval_surrogate_model import eval_surrogate_model

def find_top_k_similar_with_user_request(y_user_request, X_train, y_train, k=50):
    # euclidean distance
    distances = np.linalg.norm(y_train - y_user_request.reshape(1,-1), axis=1)
    top_k_indices = np.argsort(distances)[:k]
    return X_train[top_k_indices], y_train[top_k_indices]

def main(args, scalers=None):
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 랜덤 시드 설정
    Setting.seed_everything(args.seed)
    
    model_name = args.model # 사용할 서로게이트 모델 명
    search_model = args.search_model # 사용할 서치 모델 명명
    

    # load all data
    df = load_data(args.data_path)
    X = df.drop(columns=args.target)
    y = df[args.target]
    x_col_list = X.columns.tolist()
    X_train,y_train = X.to_numpy(),y.to_numpy() 

    # controll range
    controll_range = {}

    if scalers:
        for key, value in args.controll_range.items():
            # nominal인 경우 
            # 들어온 두 값이 같으면 하나의 값으로 제한
            # 들어온 두 값이 다르면 모든 범위 
            if type(scalers[key]).__name__ == 'LabelEncoder':
                # controll_range[key] = scalers[key].transform(np.array(value).reshape(-1,1))
                if value[0] == value[1]:
                    ran_val = scalers[key].transform(np.array(value).reshape(-1,1)).flatten()
                    controll_range[key] = (ran_val[0],ran_val[1])
                else:
                    i = x_col_list.index(key)
                    controll_range[key] = (X_train[:,i].min(),X_train[:,i].max())
            
            # continuous인 경우 
            # 들어온 값의 범위로 제한 
            else:
                controll_range[key] = tuple(scalers[key].transform(np.array(value).reshape(-1,1)).flatten())

        # target에 대한 user request에 scaler 적용 
        y_user_request = []
        for i in range(len(args.target)):
            y_user_request.append(scalers[args.target[i]].transform(np.array(args.user_request_target[i]).reshape(-1,1))[0])
        y_user_request = np.array(y_user_request).reshape(-1,1)

    else:
        raise ValueError("scalers is not provided")
    
    # user request와 유사도 높은 샘플을 추출 
    X_test,y_test = find_top_k_similar_with_user_request(y_user_request, X_train, y_train, k=5)


    # 원본 데이터 scale로 변환
    def inverse_transform(df):#, col_names):
        """
        df : scaled col이 있는 df 
        """
        df_tmp = df.copy()
        df_scaled_cols = ['_'.join(i.split('_')[2:]) for i in df.columns]
        for i in range(len(df_scaled_cols)):
            col_reshaped = df_tmp[df.columns[i]].values.reshape(-1, 1)
            inversed = scalers[df_scaled_cols[i]].inverse_transform(col_reshaped)
            df_tmp[df.columns[i]] = inversed.flatten()
        return df_tmp

    # 데이터셋 형태 출력
    logging.info(f"X_train.shape: {X_train.shape}")
    logging.info(f"X_test.shape: {X_test.shape}")
    logging.info(f"y_train.shape: {y_train.shape}")
    logging.info(f"y_test.shape: {y_test.shape}")
    
    # surrogate model load
    # classification, regression, multi-regression
    if len(args.target) > 1:
        model_load_func = getattr(surrogate, f'{model_name}_multi_load')
    else:
        if type(scalers[args.target[0]]).__name__ == 'LabelEncoder':
            unique_classes_train = np.unique(y_train)
            if len(unique_classes_train) > 10 and model_name == 'tabpfn':
                print(f'훈련 데이터의 고유 클래스 개수가 {len(unique_classes_train)}로 10개를 초과해, {model_name}을 실행할 수 없습니다. catboost classifier를 실행합니다.')
                model_name = 'catboost'
            print(f'{model_name} classifier load')
            model_load_func = getattr(surrogate, f'{model_name}_classification_load')
        else:
            model_load_func = getattr(surrogate, f'{model_name}_load')
    model = model_load_func(f'./prj/{args.prj_id}/surrogate_model/model')
    print(model)
    
    # classification, regression, multi-regression
    if len(args.target) > 1:
        predict_func = getattr(surrogate, f'{model_name}_multi_predict')
    else:
        if type(scalers[args.target[0]]).__name__ == 'LabelEncoder':
            predict_func = getattr(surrogate, f'{model_name}_classification_predict')
        else:
            predict_func = getattr(surrogate, f'{model_name}_predict')
    
    # 최적화/검색 수행
    # 샘플링한 실제 데이터를 기반으로 유저의 요구사항에 맞게, 최적화/검색 수행 
    search_func = getattr(search, f'{search_model}_search_deploy')
    start_time = time.time()
    opt_df = search_func(model, predict_func, X_train, X_test, y_test\
                         ,x_col_list, args.controll_name, args.optimize\
                            , args.importance, controll_range, scalers\
                                , y_user_request)
    end_time = time.time()
    print(f"search model 소요 시간: {end_time - start_time:.4f}초")

    # 최적화 결과 반환 
    # 예측한 결과에 대해 Surrogate model로 예측 값 생성 
    pred_input = X_test.copy()
    control_index = [i for i, v in enumerate(x_col_list) if v in args.controll_name]
    for i in range(len(args.controll_name)):
        pred_input[:,control_index[i]] = opt_df[f'pred_x_{args.controll_name[i]}']
    pred_y = predict_func(model, pred_input)

    # 예측한 결과를 원본 데이터 scale로 변환
    for i in range(len(args.target)):
        opt_df[f'pred_y_{args.target[i]}'] = pred_y[:,i]
        opt_df[f'pred_y_{args.target[i]}'] = inverse_transform(opt_df[[f'pred_y_{args.target[i]}']])
    
    # pred y의 원본 데이터를 원본 데이터 scale로 변환
    for i in range(len(args.target)):
        opt_df[f'test_y_{args.target[i]}'] = y_test[:,i]
        opt_df[f'test_y_{args.target[i]}'] = inverse_transform(opt_df[[f'test_y_{args.target[i]}']])

    # 유저의 요구사항에 맞게 최적화한 결과를 원본 데이터 scale로 변환
    for i in range(len(args.controll_name)):
        opt_df[f'pred_x_{args.controll_name[i]}'] = inverse_transform(opt_df[[f'pred_x_{args.controll_name[i]}']])
    
    # 샘플링한 실제 데이터를 원본 데이터 scale로 변환
    for i in range(len(x_col_list)):
        if type(scalers[x_col_list[i]]).__name__ == 'LabelEncoder':
            opt_df[f'test_x_{x_col_list[i]}'] = X_test[:,i].astype(int)
        else:
            opt_df[f'test_x_{x_col_list[i]}'] = X_test[:,i]
        opt_df[f'test_x_{x_col_list[i]}'] = inverse_transform(opt_df[[f'test_x_{x_col_list[i]}']])
    # opt_df['test_x_control'] = opt_df['test_x'].apply(lambda x : x[control_index])


    # UI 요구사항에 따른 df 변환 
    target_columns = [

    (f'pred_x_{col}',f'test_x_{col}')
    for col in args.controll_name
    ] + [
    (f'pred_y_{target}',f'test_y_{target}')
    for target in args.target
    ]

    df_transformed = []
    for pred_col, test_col in target_columns:
        df_transformed.append(
            {
                "column_name": test_col.split("_")[-1],
                "ground_truth": opt_df[test_col].tolist(),
                "predicted": opt_df[pred_col].tolist(),
            }
        )

    # 새로운 데이터프레임 생성
    df_result = pd.DataFrame(df_transformed)
    return df_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 학습 스크립트')
    arg = parser.add_argument
    arg('--model', '--model', '-model', type=str, default='catboost',
        choices=['catboost', 'tabpfn'], help='사용할 모델을 지정합니다 (기본값: catboost)')
    arg('--search_model', '--search_model', '-search_model', type=str, default='k_means',
        choices=['k_means'], help='사용할 검색/최적화 방법을 지정합니다 (기본값: k_means)')
    arg('--data_path', '--data_path', '-data_path', type=str, default='./data/concrete_processed.csv',
        help='데이터셋 CSV 파일 경로를 지정합니다')
    arg('--controll_name', '--controll_name', '-controll_name', type=list, default=['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age'],
        help='제어 변수 이름을 지정합니다')
    arg('--controll_range', '--controll_range', '-controll_range', type=dict, default={'cement': (102.0, 540.0),
                'slag': (0.0, 359.4),
                'ash': (0.0, 200.1),
                'water': (121.8, 247.0),
                'superplastic': (0.0, 32.2),
                'coarseagg': (801.0, 1145.0),
                'fineagg': (594.0, 992.6),
                'age': (1.0, 365.0)},
        help='제어 변수 범위를 지정합니다')
    arg('--target', '--target', '-target', type=list, default=['strength'],
        help='타겟 변수를 지정합니다')
    arg('--importance', '--importance', '-importance', type=dict, default={'cement': 1,
                'water': 2,
                },
        help='피쳐 별 중요도를 지정합니다')
    arg('--optimize', '--optimize', '-optimize', type=dict, default={'cement': 'maximize',
                'water': 'maximize',
                },
        help='피쳐 별 최적화 방향을 지정합니다')
    arg('--prj_id', '--prj_id', '-prj_id', type=int, default=42,
        help='프로젝트 아이디를 지정합니다')
    arg('--seed', '--seed', '-seed', type=int, default=42,
        help='재현성을 위한 랜덤 시드 (기본값: 42)')
    arg('--user_request_target', '--user_request_target', '-user_request_target', type=list, default=[0.0],
        help='사용자 요청 타겟 값을 지정합니다')
    args = parser.parse_args()

    main(args)
