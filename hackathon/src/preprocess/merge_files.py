import polars as pl
import logging
from typing import List
import os


def merge_files_in_chunks(file_paths: List[str], output_path: str, chunk_size: int = 10000000) -> None:
    """
    Polars를 사용하여 대규모 데이터를 청크 단위로 병합하고 저장합니다.

    :param file_paths: 병합할 파일 경로 리스트 (순서대로 제공되어야 함)
    :param output_path: 병합 결과를 저장할 파일 경로
    :param chunk_size: 청크 크기 (한 번에 로드할 log_data 행 수)
    """
    # 파일 경로 매핑
    brand_table_path = file_paths[0]
    category_table_path = file_paths[1]
    item_data_path = file_paths[2]
    log_data_path = file_paths[3]
    user_data_path = file_paths[4]

    # 파일 로드
    logging.info("Loading static tables (brand_table, category_table, item_data, user_data)")
    brand_table = pl.read_parquet(brand_table_path)
    category_table = pl.read_parquet(category_table_path)
    user_data = pl.read_parquet(user_data_path)
    item_data = pl.read_parquet(item_data_path)

    # item_data와 brand_table, category_table 병합
    logging.info("Merging item_data with brand_table and category_table")
    item_data = item_data.join(brand_table, on='brand_id', how='left')
    item_data = item_data.join(category_table, on='category_id', how='left')

    # log_data를 Polars로 청크 단위 읽기
    logging.info(f"Processing log_data in chunks of size {chunk_size}")
    log_data = pl.read_parquet(log_data_path)

    # 총 행 수 계산
    total_rows = log_data.height

    # 중간 결과를 저장할 임시 디렉토리
    temp_dir = "temp_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    # 청크 단위로 처리
    for i in range(0, total_rows, chunk_size):
        chunk = log_data.slice(i, chunk_size)  # 청크 분할
        logging.info(f"Processing rows {i} to {min(i + chunk_size, total_rows)}")

        # log_data와 user_data 병합
        merged_chunk = chunk.join(user_data, on='user_session_index', how='inner')
        
        # 병합된 결과와 item_data 병합
        merged_chunk = merged_chunk.join(item_data, on='product_id_index', how='inner')
        
        # 청크 저장
        chunk_file = os.path.join(temp_dir, f"chunk_{i // chunk_size}.parquet")
        merged_chunk.write_parquet(chunk_file)
        logging.info(f"Saved chunk {i // chunk_size} to {chunk_file}")

    # 저장된 모든 청크 병합
    logging.info("Combining all chunk files into the final output")
    chunk_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".parquet")])
    final_merged = pl.concat([pl.read_parquet(f) for f in chunk_files])
    final_merged.write_parquet(output_path)

    # 임시 파일 삭제
    for f in chunk_files:
        os.remove(f)
    os.rmdir(temp_dir)

    logging.info(f"Merged data saved to {output_path}")
