# data_loading.py
import polars as pl

def load_parquet_file(file_path: str) -> pl.DataFrame:
    """하나의 Parquet 파일을 로드합니다."""
    return pl.read_parquet(file_path)

def load_data():
    """필요한 모든 원본 데이터를 로드합니다."""
    file_paths = {
        "brand_table": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/raw/brand_table.parquet",
        "category_table": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/raw/category_table.parquet",
        "item_data": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/raw/item_data.parquet",
        "log_data": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/raw/log_data.parquet",
        "user_data": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/raw/user_data.parquet",
    }
    
    brand_df    = load_parquet_file(file_paths["brand_table"])
    category_df = load_parquet_file(file_paths["category_table"])
    item_df     = load_parquet_file(file_paths["item_data"])
    log_df      = load_parquet_file(file_paths["log_data"])
    user_df     = load_parquet_file(file_paths["user_data"])
    
    return brand_df, category_df, item_df, log_df, user_df
