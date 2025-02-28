# feature_enginerring.py
import polars as pl

def perform_feature_engineering(chunk_df: pl.DataFrame) -> pl.DataFrame:
    """
    주어진 청크 데이터에 대해 feature engineering을 수행합니다.
    
    1. regular_price 및 discount_rate 계산  
    2. 이상치 제거 (price <= 0)  
    3. 이벤트 유형별(view, purchase) 집계 및 구매율 계산  
    4. 판매량 추정 및 매출 계산  
    5. 각 제품별 discount_rate 통계 산출 후 필터링  
    6. 불필요한 열 제거
    
    최종 결과 DataFrame을 반환합니다.
    """
    # 1. regular_price 및 discount_rate 계산
    chunk_df = chunk_df.with_columns(
        pl.col("price").max().over("product_id_index").alias("regular_price")
    ).with_columns(
        ((pl.col("regular_price") - pl.col("price")) / pl.col("regular_price") * 100)
        .alias("discount_rate")
    )
    
    # 2. 유효한 price 값 필터링
    chunk_df = chunk_df.filter(pl.col("price") > 0)
    
    # 3. 이벤트별 집계: 조회(View) 및 구매(Purchase)
    view_stats = (
        chunk_df.filter(pl.col("event_type_index") == 1)
                .group_by(["product_id_index", "user_id_index"])
                .agg(pl.count().alias("view_count"))
    )
    purchase_stats = (
        chunk_df.filter(pl.col("event_type_index") == 3)
                .group_by(["product_id_index", "user_id_index"])
                .agg(pl.count().alias("purchase_count"))
    )
    
    # 조회 및 구매 데이터를 병합하여 구매율 계산
    purchase_rate = (
        view_stats.join(purchase_stats, on=["product_id_index", "user_id_index"], how="outer")
                  .fill_null(0)
                  .with_columns(
                        (pl.col("purchase_count") / (pl.col("view_count") + pl.col("purchase_count")) * 100)
                        .alias("purchase_rate")
                  )
    )
    
    chunk_df = chunk_df.join(
        purchase_rate.select(["product_id_index", "user_id_index", "purchase_rate"]),
        on=["product_id_index", "user_id_index"],
        how="left"
    ).fill_null(0)
    
    # 4. 판매량 추정 및 매출 계산 (구매 이벤트 기준)
    purchase_events = chunk_df.filter(pl.col("event_type_index") == 3)
    sales_volume = purchase_events.group_by("product_id_index").agg(
        pl.count().alias("sales_volume")
    )
    chunk_df = chunk_df.join(sales_volume, on="product_id_index", how="left").fill_null(0)
    chunk_df = chunk_df.with_columns(
        (pl.col("price") * pl.col("sales_volume")).alias("revenue")
    )
    
    # 5. discount_rate 통계 산출 및 필터링
    discount_statistics = chunk_df.group_by("product_id_index").agg([
        pl.col("discount_rate").n_unique().alias("unique_discount_count"),
        pl.col("discount_rate").std().alias("std_discount_rate"),
        pl.count().alias("observations_count")
    ])
    
    valid_products = discount_statistics.filter(
        (pl.col("observations_count") >= 10) & (pl.col("std_discount_rate") > 0.01)
    ).select("product_id_index")
    
    filtered_chunk_df = chunk_df.join(valid_products, on="product_id_index", how="inner")
    
    # 6. 불필요한 열 제거
    cleaned_chunk_df = filtered_chunk_df.drop(["sales_volume", "regular_price"])
    
    return cleaned_chunk_df
