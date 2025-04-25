import numpy as np
import pandas as pd


def get_country_from_coordinates(lat, lon, country_df):
    for _, row in country_df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']) or pd.isna(row['radius']):
            continue

        dlat = lat - row['latitude']
        dlon = lon - row['longitude']
        distance_deg = np.sqrt(dlat ** 2 + dlon ** 2)

        if distance_deg <= row['radius']:
            return row['country']

    return "Unknown Location"


def add_country_to_df(df, countries_df, lat_col_name='latitude', lon_col_name='longitude'):
    latitudes = df[lat_col_name].to_numpy()
    longitudes = df[lon_col_name].to_numpy()

    country_lats = countries_df['latitude'].to_numpy()
    country_lons = countries_df['longitude'].to_numpy()
    country_radii = countries_df['radius'].to_numpy()
    country_names = countries_df['country'].to_numpy()

    result = np.full(latitudes.shape, 'Unknown Location', dtype=object)

    for i in range(len(countries_df)):
        dlat = latitudes - country_lats[i]
        dlon = longitudes - country_lons[i]
        distances = np.sqrt(dlat ** 2 + dlon ** 2)

        within_radius = (distances <= country_radii[i]) & (result == 'Unknown Location')

        result[within_radius] = country_names[i]

    df['country'] = result
    return df
