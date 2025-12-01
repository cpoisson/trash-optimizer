
from openrouteservice import convert
import openrouteservice



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
