import pandas as pd

from src.utils import add_country_to_df


def read_countries_data():
    country_df = pd.read_csv('data/country_latitude_longitude_area_lookup.csv')
    country_df.columns = ['latitude', 'longitude', 'country', 'area', 'radius']
    return country_df


def read_and_reformat_yield_and_production_df():
    yield_df = pd.read_csv('data/Yield_and_Production_data.csv')
    combined_data = {}

    for _, row in yield_df.iterrows():
        country_year_key = f"{row['Country']}_{row['Year']}"

        if country_year_key not in combined_data:
            combined_data[country_year_key] = {
                'Country': row['Country'],
                'Year': row['Year']
            }

        item_element_key = f"{row['Item']}_{row['Element']}"
        combined_data[country_year_key][item_element_key] = row['Value']

    combined_df = pd.DataFrame(combined_data.values())

    sorted_columns = list(combined_df.columns)[:2] + sorted(list(combined_df.columns)[2:])
    combined_df = combined_df[sorted_columns]

    return combined_df


def read_canopint_inst_data(countries_df):
    df = pd.read_csv('data/CanopInt_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_esoil_tavg_data(countries_df):
    df = pd.read_csv('data/ESoil_tavg_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_land_cover_data(countries_df):
    df = pd.read_csv('data/Land_cover_percent_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_rainf_tavg_data(countries_df):
    df = pd.read_csv("data/Rainf_tavg_data.csv")
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_snowf_tavg_data(countries_df):
    df = pd.read_csv('data/Snowf_tavg_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soilmoi_inst_0_10_data(countries_df):
    df = pd.read_csv('data/SoilMoi0_10cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soilmoi_inst_10_40_data(countries_df):
    df = pd.read_csv('data/SoilMoi10_40cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soilmoi_inst_40_100_data(countries_df):
    df = pd.read_csv('data/SoilMoi40_100cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soilmoi_inst_100_200_data(countries_df):
    df = pd.read_csv('data/SoilMoi100_200cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soiltmp_inst_0_10_data(countries_df):
    df = pd.read_csv('data/SoilTMP0_10cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soiltmp_inst_10_40_data(countries_df):
    df = pd.read_csv("data/SoilTMP10_40cm_inst_data.csv")
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soiltmp_inst_40_100_data(countries_df):
    df = pd.read_csv('data/SoilTMP40_100cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_soiltmp_inst_100_200_data(countries_df):
    df = pd.read_csv('data/SoilTMP100_200cm_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_tveg_tavg_data(countries_df):
    df = pd.read_csv('data/TVeg_tavg_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df


def read_tws_inst_data(countries_df):
    df = pd.read_csv('data/TWS_inst_data.csv')
    df_with_country = add_country_to_df(df, countries_df)
    grouped_df = df_with_country.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
    return grouped_df
