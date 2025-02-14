# src/preprocess/processing_metadata.py

# 제거된 열을 기록하는 전역 리스트
removed_columns = []

def add_removed_columns(columns: list):
    """
    제거된 컬럼을 전역 리스트에 추가.
    :param columns: 제거된 컬럼 리스트
    """
    global removed_columns
    removed_columns.extend(columns)

def get_removed_columns():
    """
    제거된 컬럼 리스트 반환.
    """
    return removed_columns

def reset_metadata():
    """
    전역 리스트 초기화. (새로운 데이터셋 처리 시 사용)
    """
    global removed_columns
    removed_columns.clear()
