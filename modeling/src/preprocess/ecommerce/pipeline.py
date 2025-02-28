# pipeline.py
import os
import polars as pl
import gc
from data_loading import load_data
from data_transformation import merge_initial_data
from data_saving import split_and_save, save_parquet
from feature_engineering import perform_feature_engineering

def main():
    # 1. 데이터 로드
    brand_df, category_df, item_df, log_df, user_df = load_data()
    
    # 2. 원본 데이터 병합
    merged_raw_data = merge_initial_data(brand_df, category_df, item_df, log_df, user_df)
    
    # 3. 병합된 데이터를 청크 단위로 저장
    output_dir = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 한 청크당 약 100,000개의 고유 product_id_index 기준 분할
    chunk_file_paths = split_and_save(merged_raw_data, chunk_size=100000, output_dir=output_dir, file_prefix="split_data_chunk_")
    
    del merged_raw_data, brand_df, category_df, item_df, log_df, user_df
    gc.collect()

    # 4. 각 청크에 대해 feature engineering 수행 및 결과 저장
    engineered_chunks = []
    for chunk_path in chunk_file_paths:
        print(f"Processing chunk: {chunk_path}")
        chunk_df = pl.read_parquet(chunk_path)
        engineered_df = perform_feature_engineering(chunk_df)
        engineered_chunk_path = chunk_path.replace(".parquet", "_processed.parquet")
        save_parquet(engineered_df, engineered_chunk_path)
        engineered_chunks.append(engineered_df)

        # 각 청크 처리 후 메모리 정리 (이미 개별 처리 후 gc 호출을 하는 방식도 있음)
        del chunk_df, engineered_df
        gc.collect()
    
    # 5. (옵션) 모든 청크를 하나로 합쳐 최종 결과 저장
    final_engineered_df = pl.concat(engineered_chunks)
    final_output_path = os.path.join(output_dir, "final_engineered_data.parquet")
    save_parquet(final_engineered_df, final_output_path)
    
if __name__ == "__main__":
    main()
