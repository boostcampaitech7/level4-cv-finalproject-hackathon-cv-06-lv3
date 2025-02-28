# src/preprocess/dvm/feature_engineering.py

import fireducks.pandas as pd

def calculate_mode(df, group_by_col, target_cols):
    """
    특정 열을 기준으로 그룹화한 후, 지정된 열들의 최빈값을 계산합니다.

    :param df: 처리할 DataFrame
    :param group_by_col: 그룹화할 기준 열
    :param target_cols: 최빈값을 계산할 대상 열들의 리스트
    :return: 대상 열들의 최빈값을 포함한 DataFrame
    """
    mode_df = df.groupby(group_by_col).agg(
        {col: lambda x: x.mode()[0] if not x.mode().empty else None for col in target_cols}
    ).reset_index()
    print(f"{target_cols} 열들의 최빈값을 계산했습니다.")
    return mode_df

def merge_ad_tables(aggregated_df, ad_table1, ad_table2):
    """
    집계된 DataFrame과 두 개의 Ad_table을 'Genmodel_ID'를 기준으로 병합합니다.

    :param aggregated_df: 집계된 DataFrame
    :param ad_table1: 첫 번째 Ad_table DataFrame
    :param ad_table2: 두 번째 Ad_table DataFrame
    :return: 최종 병합된 DataFrame
    """
    # 첫 번째 Ad_table (ad_table1)과 병합
    merged = pd.merge(aggregated_df, ad_table1, on="Genmodel_ID", how="left")
    print("집계된 DataFrame과 첫 번째 Ad_table을 병합했습니다.")
    
    # 두 번째 Ad_table (ad_table2)과 병합
    merged = pd.merge(merged, ad_table2, on="Genmodel_ID", how="left")
    print("이전 병합 결과와 두 번째 Ad_table을 병합했습니다.")
    
    return merged

def add_annual_revenue(df):
    """
    'sales'와 'Entry_price'를 곱하여 'Annual_revenue' 열을 생성합니다.

    :param df: 병합된 DataFrame
    :return: 'Annual_revenue' 열이 추가된 DataFrame
    """
    df['Annual_revenue'] = df['sales'] * df['Entry_price']
    print("'Annual_revenue' 열을 생성했습니다.")
    return df
