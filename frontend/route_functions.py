
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


def get_route_matrix(df, starting_point, road_client, profile="driving-car"):
    # 1) Source = point de départ
    src_coord = [float(starting_point["lon"]), float(starting_point["lat"])]

    # 2) Destinations = liste de listes Python pures
    dest_coords = df.apply(
        lambda row: [float(row["Longitude"]), float(row["Latitude"])],
        axis=1
    ).tolist()  # ← très important : liste Python, pas Series

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

    # 3) Add to DF
    df = df.copy()
    df["distance_m"] = matrix["distances"][0]
    df["duration_s"] = matrix["durations"][0]
    return df

def get_dropoff(
    road_client,
    result_list,
    starting_point,
    prob_threshold=0.15,
    profile="driving-car",
    minimizer="distance",
    keep_top_k=100
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
    #Loop to keep the closest point while all trash classes have not yet been defined
    while len(keys_above_threshold) > 0:
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

        print(df_geoloc["Trash_class"].unique())
    if dfs:
        # Concatène tous les DataFrames et supprime les doublons
        cols_to_check = ["Trash_class", "Latitude", "Longitude", "distance_m", "duration_s"]  # les colonnes sûres
        result_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=cols_to_check)
        return [
    {"trash_type": row["Trash_class"], "lat": row["Latitude"], "lon": row["Longitude"], "distance_m": row["distance_m"], "duration_s": row["duration_s"]}
    for _, row in result_df.iterrows()
]
