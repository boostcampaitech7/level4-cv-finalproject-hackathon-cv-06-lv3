# src/preprocess/dvm/data_loading.py

import fireducks.pandas as pd

def load_data(file_paths):
    """
    여러 개의 CSV 파일을 pandas DataFrame으로 로드합니다.

    :param file_paths: 파일 식별자와 경로를 포함하는 딕셔너리
    :return: DataFrame의 딕셔너리
    """
    dataframes = {}
    for key, path in file_paths.items():
        try:
            dataframes[key] = pd.read_csv(path)
            print(f"{key} 파일을 {path}에서 성공적으로 로드했습니다.")
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {path}")
        except Exception as e:
            print(f"{key} 파일 로드 중 오류 발생: {e}")
    return dataframes
