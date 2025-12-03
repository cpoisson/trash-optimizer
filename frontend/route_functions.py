
from openrouteservice import convert
import openrouteservice
from dummy_function import DUMMY_DROP_OFF,DUMMY_START_POINT,DUMMY_PREDICT_RESULT,DUMMY_PROBABILITY_THRESHOLD,DUMMY_LIST_TRASH
from dummy_function import dummy_get_loc
from query.query import get_loc
import time
import pandas as pd

def manhattan_distance(lat1, lon1, lat2, lon2):
    """Distance de Manhattan approximative en degrés (suffisant pour un pré-filtre)."""
    return abs(lat1 - lat2) + abs(lon1 - lon2)

def to_ors(coords_dict):
    return [coords_dict["lon"], coords_dict["lat"]]

def get_route(pick_up,drop_off,client, profile='driving-car'):

    # Conversion en format ORS
    ors_coords = [to_ors(pick_up), to_ors(drop_off)]

    route = client.directions(
        coordinates=ors_coords,
        profile=profile,
        format="geojson"
    )

    return route

def get_dropoff(
    client,
    trash_dict,
    starting_point,
    prob_threshold=0.15,
    profile="driving-car",
    minimizer="distance",
    keep_top_k=100
):
    keys_above_threshold = [
    key
    for d in trash_dict
    for key, value in d.items()
    if value > prob_threshold
]
    dfs = []
    while len(keys_above_threshold) > 0:
        df_geoloc = get_loc(list_trash= keys_above_threshold)

        df_geoloc["route"] = None

    # Boucle sur les lignes
        for idx, row in df_geoloc.iterrows():
            df_geoloc.at[idx, "route"] = get_route(
                pick_up=starting_point,
                drop_off={"lat": row["Latitude"], "lon": row["Longitude"]},
                client=client,
                profile=profile
            )
            time.sleep(0.5)

        if minimizer == 'distance':
            df_geoloc["distance"] = df_geoloc["route"].apply(
            lambda route: route["features"][0]["properties"]["summary"]["distance"]
        )
            df_geoloc['unit'] = 'meter'
        elif minimizer == 'duration':
            df_geoloc["distance"] = df_geoloc["route"].apply(
        lambda route: route["features"][0]["properties"]["summary"]["duration"]
    )
            df_geoloc['unit'] = 'second'

        idx = df_geoloc["distance"].idxmin()
        df_min_per_type = df_geoloc.loc[[idx]].reset_index(drop=True)
        dfs.append(df_min_per_type)
        class_found = df_geoloc.loc[idx, "Trash_class"]
        keys_above_threshold.remove(class_found)

        print(df_geoloc["Trash_class"].unique())
    if dfs:
        # Concatène tous les DataFrames et supprime les doublons
        cols_to_check = ["Trash_class", "Latitude", "Longitude", "distance", "unit"]  # les colonnes sûres
        result_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=cols_to_check)
        return [
    {"trash_type": row["Trash_class"], "lat": row["Latitude"], "lon": row["Longitude"], "distance": row["distance"], "unit": row["unit"]}
    for _, row in result_df.iterrows()
]
