# data_transformation.py
import polars as pl

def merge_initial_data(
    brand_df: pl.DataFrame,
    category_df: pl.DataFrame,
    item_df: pl.DataFrame,
    log_df: pl.DataFrame,
    user_df: pl.DataFrame
) -> pl.DataFrame:
    """
    초기 데이터 병합:
      - 브랜드 데이터와 아이템 데이터를 brand_id 기준으로 병합
      - 카테고리 데이터를 병합 (category_id 기준)
      - 필요한 열 선택 및 간결한 category_id 생성
      - 로그 데이터와 사용자 데이터를 user_session_index 기준으로 병합
    """
    # 브랜드와 아이템 데이터 병합
    brand_item_merged = brand_df.join(item_df, on="brand_id", how="left")
    
    # 카테고리 데이터 병합
    brand_item_category_merged = brand_item_merged.join(category_df, on="category_id", how="left")
    
    selected_columns = [
        "brand", "brand_id", "product_id_index",
        "category_2", "category_3", "category_2_id", "category_3_id"
    ]
    selected_data = brand_item_category_merged.select(selected_columns)
    
    # 간결한 category_id 생성 (문자열 결합 후 UInt32 변환)
    selected_data = selected_data.with_columns(
        (pl.col("category_2_id").cast(pl.Utf8) + pl.col("category_3_id").cast(pl.Utf8))
        .cast(pl.UInt32)
        .alias("simplified_category_id")
    )
    
    # 불필요한 열 제거
    selected_data = selected_data.drop(["category_2_id", "category_3_id"])
    
    # 로그 데이터와 사용자 데이터 병합 (세션 기준)
    session_user_merged = log_df.join(user_df, on="user_session_index", how="inner")
    
    # 최종 원본 데이터 병합
    merged_raw_data = selected_data.join(session_user_merged, on="product_id_index", how="left")
    
    return merged_raw_data
