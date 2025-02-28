import fireducks.pandas as pd

def inverse_transform_control_variables(scaled_values: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """
    스케일링된 control variable 값을 원래의 값으로 복원한다.

    :param scaled_values: 스케일링된 데이터프레임 (control variables만 포함)
    :param scalers: 각 컬럼별 적용된 스케일러 객체 딕셔너리
    :return: 원래의 스케일로 복원된 데이터프레임
    """
    original_values = scaled_values.copy()

    for col in scaled_values.columns:
        if col in scalers:  # 스케일러가 존재하는 경우
            original_values[[col]] = scalers[col].inverse_transform(scaled_values[[col]])
            print(f"{col} 역변환 완료")
        else:
            print(f"{col}: 스케일러 정보 없음 (역변환 불가)")

    return original_values
