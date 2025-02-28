# src/preprocess/dvm/data_transformation.py

import fireducks.pandas as pd

def melt_sales_table(df_sales):
    """
    Sales_table을 wide 형식에서 long 형식으로 변환합니다.

    :param df_sales: Sales_table DataFrame
    :return: 변환된 Sales_table DataFrame
    """
    df_melted = df_sales.melt(
        id_vars=["Maker", "Genmodel", "Genmodel_ID"],
        var_name="Year",
        value_name="sales"
    )
    # 'Year' 열을 정수형으로 변환하여 일치 조건 충족
    df_melted["Year"] = df_melted["Year"].astype(int)
    print("Sales_table을 long 형식으로 변환했습니다.")
    return df_melted

def merge_price_sales(df_price, df_sales_melted):
    """
    Price_table과 변환된 Sales_table을 'Genmodel_ID'와 'Year'를 기준으로 병합합니다.

    :param df_price: Price_table DataFrame
    :param df_sales_melted: 변환된 Sales_table DataFrame
    :return: 병합된 DataFrame
    """
    merged = pd.merge(
        df_price,
        df_sales_melted[["Genmodel_ID", "Year", "sales"]],
        on=["Genmodel_ID", "Year"],
        how="left"
    )
    print("Price_table과 Sales_table을 병합했습니다.")
    return merged

def aggregate_sales(merged_df):
    """
    지정된 열 조합에 따라 sales 값을 합산하여 집계합니다.

    :param merged_df: 병합된 Price와 Sales DataFrame
    :return: sales가 합산된 집계된 DataFrame
    """
    aggregated = (
        merged_df.groupby(
            ["Maker", "Genmodel", "Genmodel_ID", "Year", "Entry_price"],
            as_index=False
        )
        .agg({"sales": "sum"})
    )
    print("Maker, Genmodel, Genmodel_ID, Year, Entry_price 기준으로 sales를 집계했습니다.")
    return aggregated
