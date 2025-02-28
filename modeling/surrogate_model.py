# main.py

import argparse
import logging
import time
import os 

import src.datasets as datasets
import src.search as search
import src.surrogate as surrogate
from src.utils import Setting, measure_time

import fireducks.pandas as pd
import numpy as np

# from src.surrogate.eval_surrogate_model import eval_surrogate_model

def main(args, scalers=None):
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 랜덤 시드 설정
    Setting.seed_everything(args.seed)
    
    model_name = args.model # 사용할 서로게이트 모델 명
    
    # 데이터 로드 및 분할
    load_data_func = datasets.load_and_split_data_with_x_col_list
    X_train, X_test, y_train, y_test, x_col_list = load_data_func(args.data_path, args.target)

    if model_name == 'tabpfn':
        if X_train.shape[0] > 3000:
            X_train = X_train[:3000]
            y_train = y_train[:3000]
            print("X_train.shape: ", X_train.shape)
            print("X_test.shape: ", X_test.shape)

    load_data_loader_func = getattr(datasets, f'{model_name}_load_data')

    # TODO sampling and catboost trial 조절 
    train_loader, val_loader = load_data_loader_func(X_train, X_test, y_train, y_test)
    
    # 데이터셋 형태 출력
    logging.info(f"X_train.shape: {X_train.shape}")
    logging.info(f"X_test.shape: {X_test.shape}")
    logging.info(f"y_train.shape: {y_train.shape}")
    logging.info(f"y_test.shape: {y_test.shape}")
    

    # 모델 학습
    # multi-regression, regression or classification 
    if len(args.target) > 1:
        train_func = getattr(surrogate, f'{model_name}_multi_train')
    else:
        if type(scalers[args.target[0]]).__name__ == 'LabelEncoder':
            unique_classes_train = np.unique(y_train)
            unique_classes_test = np.unique(y_test)
            if len(unique_classes_train) > 10 and model_name == 'tabpfn':
                print(f'훈련 데이터의 고유 클래스 개수가 {len(unique_classes_train)}로 10개를 초과해, {model_name}을 실행할 수 없습니다. catboost classifier를 실행합니다.')
                model_name = 'catboost'
            train_func = getattr(surrogate, f'{model_name}_classification_train')
        else:
            train_func = getattr(surrogate, f'{model_name}_train')

    model = train_func(train_loader, val_loader)
    
    # predict
    # multi-regression, regression or classification 
    if len(args.target) > 1:
        predict_func = getattr(surrogate, f'{model_name}_multi_predict')
    else:
        #check 
        if "classification" in train_func.__name__:
            predict_func = getattr(surrogate, f'{model_name}_classification_predict')
        else:
            predict_func = getattr(surrogate, f'{model_name}_predict')

    y_pred = predict_func(model, X_test)

    print(y_pred.shape)
    print(y_test.shape)

    # evaluation
    if "classification" in train_func.__name__:
        acc, prec,rec, f1, auc, logloss = surrogate.eval_classification_model(y_test, y_pred)
        df_eval = pd.DataFrame({'Accuracy': None, 'Precision': None, 'r2': acc, 'target': args.target})
    else:
        rmse, mae, r2 = surrogate.eval_surrogate_model(y_train, y_pred, y_test)
        df_eval = pd.DataFrame({'rmse': rmse, 'mae': mae, 'r2': r2, 'target': args.target})
    
    print(df_eval)

    print('y_test.shape',y_test.shape)
    print('y_pred.shape',y_pred.shape)

    os.makedirs(f'./prj/{args.prj_id}/surrogate_model', exist_ok=True)

    # save evaluation
    df_eval.to_csv(f'./prj/{args.prj_id}/surrogate_model/eval.csv', index=False)

    # ground truth and prediction + ranking row by difference
    if scalers:    
        for i in range(len(args.target)):
            df_rank = pd.DataFrame(y_test)
            df_rank['y_test'] = y_test[:,i]
            df_rank['y_pred'] = y_pred[:,i]

            if type(scalers[args.target[i]]).__name__ == 'LabelEncoder':
                df_rank['y_test'] = df_rank['y_test'].astype(int)
                df_rank['y_pred'] = df_rank['y_pred'].astype(int)
                df_rank['diff'] = (y_test[:,i] != y_pred[:,i]).astype(int)

            else:
                df_rank['y_test'] = scalers[args.target[i]].inverse_transform(df_rank['y_test'].values.reshape(-1,1)).flatten()
                df_rank['y_pred'] = scalers[args.target[i]].inverse_transform(df_rank['y_pred'].values.reshape(-1,1)).flatten()
                df_rank['diff'] = abs(df_rank['y_test'] - df_rank['y_pred'])
                
            df_rank['rank'] = df_rank['diff'].rank(method='min').astype(int)

            df_rank.to_csv(f'./prj/{args.prj_id}/surrogate_model/surrogate_model_{args.target[i]}.csv', index=False)

    # feature importance 
    if model_name == 'catboost':
        df_importance = pd.DataFrame({'feature': x_col_list, 'importance': model.get_feature_importance()/model.get_feature_importance().sum()})
        df_importance.to_csv(f'./prj/{args.prj_id}/surrogate_model/importance.csv', index=False)

    # save model
    save_model_func = getattr(surrogate, f'{model_name}_save')
    save_model_func(model, f'./prj/{args.prj_id}/surrogate_model/model')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='surrogate model 학습 스크립트')
    arg = parser.add_argument
    arg('--target', '--target', '-target', type=list, default=['strength'],
        help='타겟 변수를 지정합니다')
    arg('--data_path', '--data_path', '-data_path', type=str, default='./data/concrete_processed.csv',
        help='데이터셋 CSV 파일 경로를 지정합니다')
    arg('--model', '--model', '-model', type=str, default='catboost',
        choices=['catboost', 'tabpfn'], help='사용할 모델을 지정합니다 (기본값: catboost)')
    arg('--prj_id', '--prj_id', '-prj_id', type=int, default=42,
        help='프로젝트 아이디를 지정합니다')
    arg('--seed', '--seed', '-seed', type=int, default=42,
        help='재현성을 위한 랜덤 시드 (기본값: 42)')
    args = parser.parse_args()

    main(args)
