import numpy as np
import pandas as pd


def get_country_from_coordinates(lat, lon, country_df):
    for _, row in country_df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']) or pd.isna(row['radius']):
            continue

        dlat = lat - row['latitude']
        dlon = lon - row['longitude']
        distance_deg = np.sqrt(dlat**2 + dlon**2)

        if distance_deg <= row['radius']:
            return row['country']

    return "Unknown Location"
