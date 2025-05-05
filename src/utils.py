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

    df['Country'] = result

    df = df.drop([lat_col_name, lon_col_name], axis=1)
    if 'year' in df.columns:
        df = df.rename(columns={'year': 'Year'})

    return df


def one_hot_encode_keep_original(df, column_name, insert_after_col):
    # Create dummy variables
    dummies = pd.get_dummies(df[column_name], prefix=column_name, dtype=float)

    # Find the position to insert the dummies
    col_index = df.columns.get_loc(insert_after_col)

    # Split df into 3 parts: before, the column, after
    before = df.iloc[:, :col_index + 1]
    after = df.iloc[:, col_index + 1:]

    # Concatenate: before + dummies + after
    df_new = pd.concat([before, dummies, after], axis=1)

    return df_new
