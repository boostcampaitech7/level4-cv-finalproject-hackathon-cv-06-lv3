# data_saving.py
import os
import polars as pl

def save_parquet(df: pl.DataFrame, file_path: str):
    """주어진 DataFrame을 Parquet 파일로 저장합니다."""
    df.write_parquet(file_path)
    print(f"Saved: {file_path}")

def split_and_save(
    df: pl.DataFrame, 
    chunk_size: int, 
    output_dir: str, 
    file_prefix: str = "split_data_chunk_"
) -> list:
    """
    product_id_index 기준으로 데이터를 청크로 나누어 저장합니다.
    
    Args:
        df: 병합된 원본 DataFrame.
        chunk_size: 한 청크에 포함할 고유 product_id_index의 개수.
        output_dir: 청크 파일을 저장할 디렉토리.
        file_prefix: 저장 파일명 접두사.
    
    Returns:
        저장된 파일 경로 리스트.
    """
    unique_product_ids = df.select("product_id_index").unique().to_series()
    chunks = [
        unique_product_ids[i:i + chunk_size] 
        for i in range(0, len(unique_product_ids), chunk_size)
    ]
    saved_files = []
    
    for i, prod_id_chunk in enumerate(chunks):
        chunk_df = df.filter(pl.col("product_id_index").is_in(prod_id_chunk))
        output_file = os.path.join(output_dir, f"{file_prefix}{i + 1}.parquet")
        save_parquet(chunk_df, output_file)
        saved_files.append(output_file)
    
    return saved_files
