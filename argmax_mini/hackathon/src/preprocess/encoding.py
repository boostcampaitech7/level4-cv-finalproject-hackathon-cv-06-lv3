# src/preprocess/encoding.py
import numpy as np
import fireducks.pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer


def one_hot_encode(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    지정된 열에 대해 One-Hot Encoding을 수행하고,
    원본 열을 삭제한 뒤 인코딩된 열을 데이터프레임에 추가한다.
    """
    df = pd.get_dummies(df, columns=cols, drop_first=True)
    return df

def label_encode(df: pd.DataFrame, cols: list, scaler: dict) -> pd.DataFrame:
    """
    Label Encoding 수행. 각 열마다 LabelEncoder를 독립적으로 학습.
    """
    for col in cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        scaler[col] = le
    return df, scaler

def bert_encode(categories: list, model, tokenizer, batch_size: int = 32, max_length: int = 128) -> pd.DataFrame:
    """
    BERT 임베딩 결과를 평균값으로 축소하여 반환.
    :param categories: 범주형 데이터 리스트
    :param model: BERT 모델
    :param tokenizer: BERT 토크나이저
    :param batch_size: 배치 크기 (기본값: 32)
    :param max_length: 최대 토큰 길이 (기본값: 128)
    :return: 각 범주의 BERT 임베딩 평균값 DataFrame
    """
    if model is None or tokenizer is None:
        raise ValueError("BERT 모델 또는 토크나이저가 초기화되지 않았습니다.")

    embeddings = []
    unique_categories = list(set(categories))
    category_to_mean = {}

    try:
        for i in range(0, len(unique_categories), batch_size):
            batch = unique_categories[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
            with torch.no_grad():
                outputs = model(**inputs)
            # CLS 토큰의 벡터 평균값 계산
            cls_mean_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().mean(axis=1)
            for category, mean_embedding in zip(batch, cls_mean_embeddings):
                category_to_mean[category] = mean_embedding
    except Exception as e:
        print(f"BERT 임베딩 수행 중 오류 발생: {e}")
        for category in unique_categories:
            category_to_mean[category] = 0.0  # 오류 발생 시 평균값을 0으로 대체

    for category in categories:
        embeddings.append(category_to_mean.get(category, 0.0))

    return pd.DataFrame(embeddings, index=categories)
