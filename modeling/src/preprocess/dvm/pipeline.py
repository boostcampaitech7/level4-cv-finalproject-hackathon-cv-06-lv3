# src/preprocess/dvm/pipeline.py

import fireducks.pandas as pd
from .data_loading import load_data
from .data_transformation import melt_sales_table, merge_price_sales, aggregate_sales
from .feature_engineering import calculate_mode, merge_ad_tables, add_annual_revenue
from .data_saving import save_final_dataframe

def preprocess_dvm_dataset():
    """
    DVM 데이터셋 병합 및 Feature Engineering 전 과정을 실행하는 파이프라인 함수입니다.
    """
    # 파일 경로 정의
    file_paths = {
        "Basic_table": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm/Basic_table.csv",
        "Price_table": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm/Price_table.csv",
        "Sales_table": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm/Sales_table.csv",
        "Trim_table": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm/Trim_table.csv",
        "Ad_table1": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm/Ad_table.csv",
        "Image_table": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm/Image_table.csv",
        "Ad_table2": "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm/Ad_table (extra).csv"
    }

    # 모든 데이터셋 로드
    data = load_data(file_paths)

    # 필수 DataFrame이 로드되었는지 확인
    essential_keys = ["Price_table", "Sales_table", "Ad_table1", "Ad_table2"]
    for key in essential_keys:
        if key not in data:
            print(f"필수 DataFrame이 누락되었습니다: {key}. 프로세스를 종료합니다.")
            return

    # Sales_table을 long 형식으로 변환
    df_sales_melted = melt_sales_table(data["Sales_table"])

    # Price_table과 변환된 Sales_table 병합
    merged_price_sales = merge_price_sales(data["Price_table"], df_sales_melted)

    # sales 집계
    aggregated_sales = aggregate_sales(merged_price_sales)

    # 첫 번째 Ad_table (Ad_table1) 처리
    ad1_filtered = data["Ad_table1"][["Genmodel_ID", "Bodytype", "Engin_size", "Gearbox", "Fuel_type"]]
    ad1_mode = calculate_mode(ad1_filtered, "Genmodel_ID", ["Bodytype", "Engin_size", "Gearbox", "Fuel_type"])

    # 두 번째 Ad_table (Ad_table2) 처리
    # 열 이름의 공백 제거
    data["Ad_table2"].columns = data["Ad_table2"].columns.str.strip()
    ad2_filtered = data["Ad_table2"][
        ["Genmodel_ID", "Engine_power", "Wheelbase", "Height", "Width", "Length", 
         "Average_mpg", "Top_speed", "Seat_num", "Door_num"]
    ]
    ad2_mode = calculate_mode(ad2_filtered, "Genmodel_ID", [
        "Engine_power", "Wheelbase", "Height", "Width", "Length", 
        "Average_mpg", "Top_speed", "Seat_num", "Door_num"
    ])

    # 집계된 sales와 두 Ad_table 병합
    merged_final = merge_ad_tables(aggregated_sales, ad1_mode, ad2_mode)

    # Annual Revenue 계산
    final_df = add_annual_revenue(merged_final)

    # 최종 DataFrame을 CSV 파일로 저장
    output_file_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/dvm_merged_df.csv"
    save_final_dataframe(final_df, output_file_path)

if __name__ == "__main__":
    preprocess_dvm_dataset()
