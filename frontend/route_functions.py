
from openrouteservice import convert
import openrouteservice
from dummy_function import DUMMY_DROP_OFF,DUMMY_START_POINT,DUMMY_PREDICT_RESULT,DUMMY_PROBABILITY_THRESHOLD,DUMMY_LIST_TRASH
from dummy_function import dummy_get_loc

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

def get_dropoff(client,trash_dict= DUMMY_PREDICT_RESULT,starting_point = DUMMY_START_POINT,prob_threshold = DUMMY_PROBABILITY_THRESHOLD, profile='driving-car',minimizer='distance'):
    keys_above_threshold = [
    key
    for d in trash_dict
    for key, value in d.items()
    if value > prob_threshold
]
    df_geoloc = dummy_get_loc(list_trash= keys_above_threshold)

    df_geoloc["route"] = df_geoloc.apply(
        lambda row: get_route(
            pick_up=starting_point,
            drop_off={"lat": row["lat"], "lon": row["lon"]},
            client=client,
            profile=profile
        ),
        axis=1
    )
    if minimizer == 'distance':
        df_geoloc["distance"] = df_geoloc["route"].apply(
        lambda route: route["features"][0]["properties"]["summary"]["distance"]
    )
    elif minimizer == 'duration':
        df_geoloc["distance"] = df_geoloc["route"].apply(
    lambda route: route["features"][0]["properties"]["summary"]["duration"]
)

    idx = df_geoloc.groupby("trash_type")["distance"].idxmin()
    df_min_per_type = df_geoloc.loc[idx].reset_index(drop=True)
    return df_min_per_type
