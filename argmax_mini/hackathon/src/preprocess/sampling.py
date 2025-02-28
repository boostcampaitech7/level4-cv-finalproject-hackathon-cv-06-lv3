import fireducks.pandas as pd

def sample_dataframe(df: pd.DataFrame, sample_size: int = 1_000_000, random_state: int = 42) -> pd.DataFrame:
    """
    데이터프레임에서 일부 샘플을 랜덤으로 추출하는 함수.
    
    Args:
        df (pd.DataFrame): 전체 데이터프레임
        sample_size (int): 샘플링할 행 개수 (기본값: 1,000,000)
        random_state (int): 랜덤 시드 (기본값: 42)

    Returns:
        pd.DataFrame: 샘플링된 데이터프레임
    """
    return df.sample(n=min(len(df), sample_size), random_state=random_state)
