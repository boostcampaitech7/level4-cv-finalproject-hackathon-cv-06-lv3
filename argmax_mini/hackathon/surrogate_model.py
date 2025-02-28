# main.py

import argparse
import logging
import time
import os

import fireducks.pandas as pd
import numpy as np

import hackathon.src.datasets as datasets
import hackathon.src.search as search
import hackathon.src.surrogate as surrogate
from hackathon.src.utils import Setting, measure_time
# from src.surrogate.eval_surrogate_model import eval_surrogate_model


def main(args, scalers=None):
    # 로깅 설정
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 랜덤 시드 설정
    Setting.seed_everything(args.seed)

    model_name = args.model  # 사용할 서로게이트 모델 명

    # 데이터 로드 및 분할
    load_data_func = datasets.load_and_split_data_with_x_col_list
    X_train, X_test, y_train, y_test, x_col_list = load_data_func(
        args.data_path, args.target)

    if model_name == 'tabpfn':
        if X_train.shape[0] > 3000:
            X_train = X_train[:3000]
            y_train = y_train[:3000]

    load_data_loader_func = getattr(datasets, f'{model_name}_load_data')

    train_loader, val_loader = load_data_loader_func(
        X_train, X_test, y_train, y_test)

    # 모델 학습
    if len(args.target) > 1:
        train_func = getattr(surrogate, f'{model_name}_multi_train')
    else:
        if type(scalers[args.target[0]]).__name__ == 'LabelEncoder':
            unique_classes_train = np.unique(y_train)
            unique_classes_test = np.unique(y_test)
            if len(unique_classes_train) > 10 and model_name == 'tabpfn':
                print(
                    f'훈련 데이터의 고유 클래스 개수가 {len(unique_classes_train)}로 10개를 초과해, {model_name}을 실행할 수 없습니다. catboost classifier를 실행합니다.')
                model_name = 'catboost'
            train_func = getattr(
                surrogate, f'{model_name}_classification_train')
        else:
            train_func = getattr(surrogate, f'{model_name}_train')

    model = train_func(train_loader, val_loader)

    if len(args.target) > 1:
        predict_func = getattr(surrogate, f'{model_name}_multi_predict')
    else:
        predict_func = getattr(surrogate, f'{model_name}_predict')
    y_pred = predict_func(model, X_test)

    if "classification" in train_func.__name__:
        acc, prec, rec, f1, auc, logloss = surrogate.eval_classification_model(
            y_test, y_pred)
        df_eval = pd.DataFrame(
            {'rmse': None, 'mae': None, 'r2': acc, 'target': args.target})
    else:
        rmse, mae, r2 = surrogate.eval_surrogate_model(y_train, y_pred, y_test)
        df_eval = pd.DataFrame(
            {'rmse': rmse, 'mae': mae, 'r2': r2, 'target': args.target})

    if scalers:
        all_rank = []
        for i in range(len(args.target)):
            df_rank = pd.DataFrame(y_test)
            df_rank['y_test'] = y_test[:, i]
            df_rank['y_pred'] = y_pred[:, i]
            if type(scalers[args.target[i]]).__name__ == 'LabelEncoder':
                df_rank['y_test'] = df_rank['y_test'].astype(int)
                df_rank['y_pred'] = df_rank['y_pred'].astype(int)
                df_rank['diff'] = (y_test[:, i] != y_pred[:, i]).astype(int)

            df_rank['y_test'] = scalers[args.target[i]].inverse_transform(
                df_rank['y_test'].values.reshape(-1, 1)).flatten()
            df_rank['y_pred'] = scalers[args.target[i]].inverse_transform(
                df_rank['y_pred'].values.reshape(-1, 1)).flatten()

            if type(scalers[args.target[i]]).__name__ != 'LabelEncoder':
                df_rank['diff'] = abs(df_rank['y_test'] - df_rank['y_pred'])

            df_rank['column_name'] = args.target[i]

            top5 = df_rank.nsmallest(5, 'diff')
            bottom5 = df_rank.nlargest(5, 'diff')

            # Combine the two subsets.
            df_subset = pd.concat([top5, bottom5])
            df_subset['rank'] = df_subset['diff'].rank()
            all_rank.append(df_subset)

        df_rank = pd.concat(all_rank)

    os.makedirs(f'./temp/surrogate_model', exist_ok=True)

    save_model_func = getattr(surrogate, f'{model_name}_save')
    model_path = save_model_func(model, f'./temp/surrogate_model/model')

    if model_name == 'catboost':
        df_importance = pd.DataFrame({'feature': x_col_list, 'importance': model.get_feature_importance(
        )/model.get_feature_importance().sum()})

        return df_rank, df_eval, df_importance, model_path
    else:
        return df_rank, df_eval, model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 학습 스크립트')
    arg = parser.add_argument
    # arg('--dataset', '--dset', '-dset', type=str, default='cement',
    #     choices=['cement', 'melb', 'car'], help='사용할 데이터셋을 지정합니다 추후 제거')
    arg('--target', '--target', '-target', type=list, default=['strength'],
        help='타겟 변수를 지정합니다')
    arg('--data_path', '--data_path', '-data_path', type=str, default='./data/concrete_processed.csv',
        help='데이터셋 CSV 파일 경로를 지정합니다')
    arg('--model', '--model', '-model', type=str, default='lightgbm',
        choices=['lightgbm', 'catboost', 'tabpfn'], help='사용할 모델을 지정합니다 (기본값: lightgbm)')
    arg('--flow_id', '--flow_id', '-flow_id', type=int, default=42,
        help='플로우 아이디를 지정합니다')
    arg('--seed', '--seed', '-seed', type=int, default=42,
        help='재현성을 위한 랜덤 시드 (기본값: 42)')
    args = parser.parse_args()

    # TODO: omegaconf 적용

    main(args)
