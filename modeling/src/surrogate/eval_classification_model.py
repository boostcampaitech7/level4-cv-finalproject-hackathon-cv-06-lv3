from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


def eval_classification_model(y_true, y_pred, y_prob=None, average="macro"):
    """
    분류 모델을 평가하는 함수

    Args:
        y_true (array-like): 실제 레이블 (0 또는 1, 혹은 다중 클래스 정답)
        y_pred (array-like): 예측된 레이블 (0 또는 1, 혹은 다중 클래스 예측값)
        y_prob (array-like, optional): 예측 확률 값 (AUC 계산을 위해 필요, 기본값: None)
        average (str, optional): 다중 클래스 평가 시 평균 방식 선택 ('macro', 'micro', 'weighted'). 기본값은 'macro'.

    Returns:
        float: 정확도 (Accuracy)
        float: 정밀도 (Precision)
        float: 재현율 (Recall)
        float: F1-score
        float: ROC AUC 점수 (y_prob이 제공된 경우에만 계산)
        float: 로그 손실 (y_prob이 제공된 경우에만 계산)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average)

    auc = None
    if (
        y_prob is not None and len(set(y_true)) > 1
    ):  # AUC는 2개 이상의 클래스가 있어야 계산 가능
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)

    logloss = None
    if y_prob is not None:
        logloss = log_loss(y_true, y_prob)

    return acc, prec, rec, f1, auc, logloss
