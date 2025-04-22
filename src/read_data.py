import pandas as pd


def read_countries_data():
    country_df = pd.read_csv('../data/country_latitude_longitude_area_lookup.csv')
    country_df.columns = ['latitude', 'longitude', 'country', 'area', 'radius']
    return country_df
