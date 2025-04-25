import pandas as pd

from src.utils import add_country_to_df


def read_countries_data():
    country_df = pd.read_csv('../data/country_latitude_longitude_area_lookup.csv')
    country_df.columns = ['latitude', 'longitude', 'country', 'area', 'radius']
    return country_df


def read_canopint_inst_data(countries_df):
    df = pd.read_csv('../data/CanopInt_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_esoil_tavg_data(countries_df):
    df = pd.read_csv('../data/ESoil_tavg_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_land_cover_data(countries_df):
    df = pd.read_csv('../data/Land_cover_percent_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_rainf_tavg_data(countries_df):
    df = pd.read_csv("../data/Rainf_tavg_data.csv")
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_snowf_tavg_data(countries_df):
    df = pd.read_csv('../data/Snowf_tavg_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soilmoi_inst_0_10_data(countries_df):
    df = pd.read_csv('../data/SoilMoi0_10cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soilmoi_inst_10_40_data(countries_df):
    df = pd.read_csv('../data/SoilMoi10_40cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soilmoi_inst_40_100_data(countries_df):
    df = pd.read_csv('../data/SoilMoi40_100cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soilmoi_inst_100_200_data(countries_df):
    df = pd.read_csv('../data/SoilMoi100_200cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soiltmp_inst_0_10_data(countries_df):
    df = pd.read_csv('../data/SoilTMP0_10cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soiltmp_inst_10_40_data(countries_df):
    df = pd.read_csv("../data/SoilTMP10_40cm_inst_data.csv")
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soiltmp_inst_40_100_data(countries_df):
    df = pd.read_csv('../data/SoilTMP40_100cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_soiltmp_inst_100_200_data(countries_df):
    df = pd.read_csv('../data/SoilTMP100_200cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_tveg_tavg_data(countries_df):
    df = pd.read_csv('../data/TVeg_tavg_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country


def read_tws_inst_data(countries_df):
    df = pd.read_csv('../data/TWS_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    return df_with_country
