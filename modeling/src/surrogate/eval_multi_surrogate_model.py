import numpy as np


def eval_multi_surrogate_model(model, y_train, y_pred, y_test, target_cols):
    """
    다중 타겟 회귀 모델의 성능을 평가하는 함수

    Args:
        model (model): 학습된 모델 객체
        y_train (numpy.ndarray): 학습 데이터의 실제 타겟 값
        y_pred (numpy.ndarray): 테스트 데이터에 대한 모델의 예측 값
        y_test (numpy.ndarray): 테스트 데이터의 실제 타겟 값
        target_cols (list of str): 예측 대상 변수(타겟)의 컬럼명 리스트

    Returns:
        float: 평균 제곱근 오차 (RMSE)의 Mean값
        float: 평균 절대 오차 (MAE)의 Mean값
        float: 결정 계수 (R² Score)의 Mean값
    """

    # RMSE, MAE, R2 값을 저장할 리스트 초기화
    rmse_list, mae_list, r2_list = [], [], []

    # 각 타겟 컬럼별 평가 수행
    for idx, col in enumerate(target_cols):
        y_true_i = y_test[:, idx]  # 실제 타겟 값
        y_pred_i = y_pred[:, idx]  # 모델 예측 값

        # MAE (Mean Absolute Error) 계산
        mae_i = np.mean(np.abs(y_true_i - y_pred_i))

        # MSE (Mean Squared Error) 계산
        mse_i = np.mean((y_true_i - y_pred_i) ** 2)

        # RMSE (Root Mean Squared Error) 계산
        rmse_i = np.sqrt(mse_i)

        # R2 Score 계산
        sse_i = np.sum((y_true_i - y_pred_i) ** 2)  # Sum of Squared Errors
        sst_i = np.sum((y_true_i - np.mean(y_train)) ** 2)  # Total Sum of Squares
        r2_i = 1 - (sse_i / sst_i)  # R-squared 값 계산

        # 각 성능 지표를 리스트에 저장
        rmse_list.append(rmse_i)
        mae_list.append(mae_i)
        r2_list.append(r2_i)

        # 개별 타겟 컬럼에 대한 평가 결과 출력
        print(f"Target '{col}' - RMSE: {rmse_i:.4f}, MAE: {mae_i:.4f}, R2: {r2_i:.4f}")

    # 전체 타겟 컬럼에 대한 평균 성능 지표 계산
    rmse_mean = np.mean(rmse_list)
    mae_mean = np.mean(mae_list)
    r2_mean = np.mean(r2_list)

    # 평균 성능 지표 출력
    print(
        f"[Average Metrics] RMSE: {rmse_mean:.4f}, MAE: {mae_mean:.4f}, R2: {r2_mean:.4f}"
    )

    # RMSE, MAE, R2의 평균 값 반환
    return rmse_mean, mae_mean, r2_mean
