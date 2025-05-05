import numpy as np
import pandas as pd

from src.utils import one_hot_encode_keep_original


def merge_data(
        yield_df,
        canopint_df,
        esoil_df,
        land_df,
        rainf_df,
        snowf_df,
        soilmoi_0_10_df,
        soilmoi_10_40_df,
        soilmoi_40_100_df,
        soilmoi_100_200_df,
        soiltmp_0_10_df,
        soiltmp_10_40_df,
        soiltmp_40_100_df,
        soiltmp_100_200_df,
        tveg_df,
        tws_df
):
    df = yield_df.copy()

    df = pd.merge(df, canopint_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, esoil_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, land_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, rainf_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, snowf_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soilmoi_0_10_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soilmoi_10_40_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soilmoi_40_100_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soilmoi_100_200_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soiltmp_0_10_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soiltmp_10_40_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soiltmp_40_100_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, soiltmp_100_200_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, tveg_df, on=['Country', 'Year'], how='left')
    df = pd.merge(df, tws_df, on=['Country', 'Year'], how='left')

    return df


def scale_data(merged_df):
    exclude_cols = ['Country', 'Year']
    cols_to_scale = [col for col in merged_df.columns if col not in exclude_cols]

    df_scaled = merged_df.copy()

    all_values = df_scaled[cols_to_scale].values.flatten()
    global_min = np.nanmin(all_values)
    global_max = np.nanmax(all_values)

    if global_max == global_min:
        raise ValueError("Global max and min are the same. Cannot perform scaling.")

    df_scaled[cols_to_scale] = (df_scaled[cols_to_scale] - global_min) / (global_max - global_min)

    scaler = {'global_min': global_min, 'global_max': global_max}

    return df_scaled, scaler


def unscale_data(scaled_array, scaler):
    global_min = scaler['global_min']
    global_max = scaler['global_max']

    unscaled_array = scaled_array * (global_max - global_min) + global_min
    return unscaled_array


def impute_data(scaled_df):
    product_cols = [
        col
        for col in scaled_df.columns
        if col.endswith('_Yield') or col.endswith('_Production')
    ]
    scaled_df[product_cols] = scaled_df.groupby('Country')[product_cols].transform(
        lambda group: group.fillna(group.mean()))
    scaled_df[product_cols] = scaled_df[product_cols].fillna(0)

    supporting_cols = [
        col
        for col in scaled_df.columns
        if col not in {'Country', 'Year'} and not col.endswith('_Yield') and not col.endswith('_Production')
    ]
    scaled_df[supporting_cols] = scaled_df[supporting_cols].interpolate(method='linear', axis=0)
    return scaled_df


def add_encoded_columns(imputed_df):
    encoded_df = one_hot_encode_keep_original(imputed_df, 'Country', 'Year')
    return encoded_df
