import fireducks.pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 데이터 불러오기 (CSV & Parquet 지원)
def load_data(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"지원되지 않는 파일 형식입니다: {file_extension}")
    
    return df

# 2. 학습 및 테스트 데이터 분할
def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
