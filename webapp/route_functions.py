
from openrouteservice import convert
import openrouteservice
from dummy_function import DUMMY_DROP_OFF,DUMMY_START_POINT,DUMMY_PREDICT_RESULT,DUMMY_PROBABILITY_THRESHOLD,DUMMY_LIST_TRASH
from dummy_function import dummy_get_loc
from query.query import get_loc
import time
import pandas as pd
import math

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


def get_route_matrix(df, starting_point, road_client, profile="driving-car"):
    # 1) Source = point de départ
    src_coord = [float(starting_point["lon"]), float(starting_point["lat"])]

    # 2) Destinations = liste de listes Python pures
    dest_coords = []
    valid_indices = []

    for idx, row in df.iterrows():
        lon_raw = row.get("Longitude", row.get("lon"))
        lat_raw = row.get("Latitude", row.get("lat"))
        if lon_raw is None or lat_raw is None:
            print(f"Ligne {idx} ignorée, coordonnées manquantes : {lon_raw}, {lat_raw}")
            continue
        try:
            lon = float(lon_raw)
            lat = float(lat_raw)
            dest_coords.append([lon, lat])
            valid_indices.append(idx)
        except (ValueError, TypeError) as e:
            print(f"Ligne {idx} ignorée, coordonnées invalides : {lon_raw}, {lat_raw} ({e})")

    if not dest_coords:
        return df

    # 3) Locations = source + destinations
    locations = [src_coord] + dest_coords

    # 4) Appel matrix ORS — SANS "format="
    matrix = road_client.distance_matrix(
        locations=locations,
        profile=profile,
        metrics=["distance", "duration"],
        sources=[0],                                  # index du point source
        destinations=list(range(1, len(locations)))  # index des destinations
    )

    # 5) Add to DF, aligning distances to valid rows only
    df = df.copy()
    df["distance_m"] = None
    df["duration_s"] = None
    distances = matrix.get("distances", [])
    durations = matrix.get("durations", [])

    if distances and durations:
        for pos, df_idx in enumerate(valid_indices):
            df.at[df_idx, "distance_m"] = distances[0][pos]
            df.at[df_idx, "duration_s"] = durations[0][pos]

    return df.dropna(subset=["distance_m", "duration_s"])

def get_dropoff(
    road_client,
    result_list,
    starting_point,
    prob_threshold=0.15,
    profile="driving-car",
    minimizer="distance",
    keep_top_k=20,
    progress_callback=None
):
    #Get all Trash Class that are "worth" exploring based on the proba threshold
    keys_above_threshold = [
        item["class"]
        for item in result_list
        if item["confidence"] > prob_threshold
    ]
    if not result_list:
        return []
    dfs = []
    i = 0
    total = len(keys_above_threshold)
    #Loop to keep the closest point while all trash classes have not yet been defined
    while len(keys_above_threshold) > 0:
        print(f" where are checking the following classes{keys_above_threshold}")
        df_geoloc = get_loc(list_trash= keys_above_threshold).copy()
        df_geoloc["manhattan"] = df_geoloc.apply(
            lambda row: manhattan_distance(
                starting_point["lat"], starting_point["lon"],
                row["Latitude"], row["Longitude"]
            ),
            axis=1
        )
    #Only keep 100 points close to the last point to check based on manhattan distance to avoid API overkill
        df_geoloc = (
            df_geoloc
            .groupby("Trash_class", group_keys=False)
            .apply(lambda g: g.nsmallest(keep_top_k, "manhattan"))
        )

        df_geoloc = get_route_matrix(df_geoloc, starting_point, road_client, profile)
        if minimizer == "distance":
            idx = df_geoloc["distance_m"].idxmin()
        else:
            idx = df_geoloc["duration_s"].idxmin()
        df_min_per_type = df_geoloc.loc[[idx]].reset_index(drop=True)
        dfs.append(df_min_per_type)
        class_found = df_geoloc.loc[idx, "Trash_class"]
        keys_above_threshold.remove(class_found)
        starting_point ={"lat": df_min_per_type["Latitude"],
                      "lon":df_min_per_type["Longitude"],
                      "trash_type":"User Start Point",
                      "distance": 0}
        if progress_callback:
            progress_callback((i + 1) / total)
            i = i+1
    if dfs:
        # Concatène tous les DataFrames et supprime les doublons
        cols_to_check = ["Trash_class", "Latitude", "Longitude", "distance_m", "duration_s"]  # les colonnes sûres
        result_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=cols_to_check)
        return [
    {"trash_type": row["Trash_class"], "lat": row["Latitude"], "lon": row["Longitude"], "distance_m": row["distance_m"], "duration_s": row["duration_s"], "address": row.get("Address", ""), "name": row.get("Name", "")}
    for _, row in result_df.iterrows()
]
