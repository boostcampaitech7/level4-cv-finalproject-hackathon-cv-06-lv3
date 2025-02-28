import fireducks.pandas as pd
from transformers import BertTokenizer, BertModel
from hackathon.src.preprocess.encoding import label_encode, bert_encode



def dynamic_encode(df: pd.DataFrame, feature_info: dict, scaler: dict) -> pd.DataFrame:
    """
    동적으로 범주형 데이터를 처리.
    - 'text' 타입의 컬럼은 BERT 임베딩 평균값 적용
    - 'categorical' 타입의 컬럼은 Label Encoding 적용
    :param df: 입력 데이터프레임
    :param feature_info: detect_features에서 반환된 컬럼 타입 정보
    :return: 인코딩된 데이터프레임
    """
    # BERT 모델 및 토크나이저 로드 (필요 시)
    bert_model, bert_tokenizer = None, None
    if 'text' in feature_info and feature_info['text']:
        try:
            print("BERT 토크나이저 및 모델 로드 중...")
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            bert_model.eval()  # 평가 모드로 전환
            print("BERT 모델 및 토크나이저 로드 완료.")
        except Exception as e:
            raise RuntimeError(f"BERT 모델 로드 중 오류 발생: {e}")

    # 텍스트 데이터 처리
    if 'text' in feature_info:
        for col in feature_info['text']:
            print(f"{col}: BERT 임베딩 평균값 적용 중...")

            # 고유값 추출 및 문자열 변환
            unique_values = df[col].unique().astype(str)
            print(f"{col}: 고유값 개수 = {len(unique_values)}")

            # BERT 임베딩 평균값 계산
            try:
                bert_mean_embeddings = bert_encode(
                    unique_values, bert_model, bert_tokenizer)
                bert_mean_embeddings.columns = [
                    f"{col}_bert_mean"]  # 단일 평균값 컬럼 추가
                bert_mean_embeddings.index = bert_mean_embeddings.index.astype(
                    str)  # 인덱스 타입 통일

                # 원본 데이터와 병합 후 컬럼 드롭
                df = df.merge(
                    bert_mean_embeddings, left_on=col, right_index=True, how='left'
                ).drop(columns=[col])
                print(f"{col}: BERT 임베딩 평균값 적용 완료.")
            except Exception as e:
                raise RuntimeError(f"{col}: BERT 임베딩 평균값 적용 중 오류 발생: {e}")

    # 일반 범주형 데이터 처리 (Label Encoding으로 통일)
    if 'categorical' in feature_info:
        for col in feature_info['categorical']:
            print(f"{col}: Label Encoding 적용 (범주형 데이터)")
            df, scaler = label_encode(df, [col], scaler)
            print(f"{col}: Label Encoding 적용 완료.")

            # numerical_categorical 데이터 처리: 원본 값을 그대로 유지하고 스케일러 정보 저장
    if 'numerical_categorical' in feature_info:
        for col in feature_info['numerical_categorical']:
            print(f"{col}: 원본 값 그대로 유지 (numerical_categorical)")

    return df, scaler
