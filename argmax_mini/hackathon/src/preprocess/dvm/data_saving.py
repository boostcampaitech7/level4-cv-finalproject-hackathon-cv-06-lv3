# src/preprocess/dvm/data_saving.py

import fireducks.pandas as pd

def save_final_dataframe(df, output_path):
    """
    최종 DataFrame을 CSV 파일로 저장합니다.

    :param df: 최종 병합된 DataFrame
    :param output_path: 저장할 CSV 파일의 경로
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"파일이 성공적으로 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
