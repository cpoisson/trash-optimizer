import streamlit as st
from PIL import Image
from dummy_function import dummy_predict, dummy_get_drop_off
from dummy_function import DUMMY_START_POINT,DUMMY_PREDICT_RESULT,DUMMY_PROBABILITY_THRESHOLD
import pydeck as pdk
import pandas as pd
import requests
from openrouteservice import convert
import openrouteservice
from route_functions import get_route,get_dropoff
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GEO_SERVICE_API_KEY = os.getenv("GEO_SERVICE_API_KEY")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL")

# Raise value errors if environment variables are not set
if not GEO_SERVICE_API_KEY:
    raise ValueError("GEO_SERVICE_API_KEY is not set in environment variables.")
if not INFERENCE_SERVICE_URL:
    raise ValueError("INFERENCE_SERVICE_URL is not set in environment variables.")

# Streamlit app
st.title("Trash-optimizer Front")

st.write("### Where do you want to pick it from ?")
address = st.text_input("Enter your adress :",
    value="Nantes")
if address:
    geolocator = Nominatim(user_agent="streamlit_app")
    try:
        location = geolocator.geocode(address)
        if location:
            st.success(f"Adresse localisée : latitude {location.latitude}, longitude {location.longitude}")
            user_input = location.latitude, location.longitude
        else:
            st.error("Adresse introuvable. Vérifiez l'orthographe.")
    except Exception as e:
            st.error(f"Erreur lors du géocodage : {e}")
road_mode = st.radio('Select a your drive mode', ('car','bike','foot'))
if road_mode == 'car':
    final_road_mode = "driving-car"
elif road_mode == 'bike':
    final_road_mode = "cycling-regular"
elif road_mode == 'foot':
    final_road_mode = "foot-walking"

minimizer = st.radio('Do you want to minimize:', ('distance','duration'))
# Step 1 — Let user choose how to provide the image
choice = st.radio(
    "Choose how to provide an image:",
    ("Upload a file", "Take a picture")
)

uploaded_file = None
camera_image = None
img = None

# Step 2 — Display the appropriate input widget
if choice == "Upload a file":
    uploaded_file = st.file_uploader(
        "Send an image file",
        type=["jpg", "png", "jpeg"]
    )
    if uploaded_file:
        img = Image.open(uploaded_file)

elif choice == "Take a picture":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        img = Image.open(camera_image)

# Step 3 — Unified output once the image exists
if img and user_input:
    st.image(img, caption="Image received")
    st.success("Image loaded!")

    # Crée un conteneur vide
    status = st.empty()

    # Bouton
    if st.button("Launch the prediction"):
        status.info("Running prediction...")  # affichage temporaire
        if uploaded_file:
            files = {"file": uploaded_file.getvalue()}  # envoie l'image brute
        elif camera_image:
            files = {"file": camera_image.getvalue()}  # envoie l'image brute
        # Si ton API attend plutôt un form-data multipart
        response = requests.post(f"{INFERENCE_SERVICE_URL}/predict", files=files)
        result_list = response.json()

        # Calcul de la prédiction
        # result_dict = dummy_predict()

        # Mise à jour du conteneur avec le résultat
        status.success("Prediction done!")  # change le message
        for i, item in enumerate(result_list):
                    st.subheader(f"Prediction {i+1}")
                    st.write(item)

        client = openrouteservice.Client(key=GEO_SERVICE_API_KEY)
        pick_up = {"lat": user_input[0],
                      "lon":user_input[1],
                      "trash_type":"User Start Point",
                      "distance": 0}
        drop_off_list = get_dropoff(client=client,
                                    trash_dict= DUMMY_PREDICT_RESULT,
                                    starting_point = pick_up,
                                    prob_threshold = DUMMY_PROBABILITY_THRESHOLD,
                                    profile=final_road_mode,
                                    minimizer=minimizer)


        all_routes = []
        all_paths = []
        all_points = []

        all_points.append({
                "lon": pick_up["lon"],
                "lat": pick_up["lat"],
                "name": "Start",
                "distance":0,
                "unit":" "
            })

        for drop_off in drop_off_list:
            # Calcul de route
            route = get_route(all_points[-1], drop_off, client)
            coords = route["features"][0]["geometry"]["coordinates"]

            # Stockage du chemin
            all_paths.append({"trash_type": drop_off["trash_type"], "path": coords})

            # Stockage des points (départ + arrivée)

            all_points.append({
                "lon": drop_off["lon"],
                "lat": drop_off["lat"],
                "name": drop_off["trash_type"],
                "distance":drop_off["distance"],
                "unit":drop_off["unit"]
            })

        # DataFrames finaux
        df_path = pd.DataFrame(all_paths)
        df_points = pd.DataFrame(all_points)
                # Layer : points
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            df_points,
            get_position='[lon, lat]',
            get_color='[200, 30, 0]',
            get_radius=20,
        )
        # Text Layer pour les libellés sur les points
        text_layer = pdk.Layer(
        "TextLayer",
        df_points,
        pickable=True,
        get_position='[lon, lat]',
        get_text="name",
        get_color=[255, 255, 0],  # JAUNE 🔥
        get_size=20,
        get_alignment_baseline="'bottom'"
    )
        # Layer : la route (pas une ligne droite !)
        route_layer = pdk.Layer(
            "PathLayer",
            df_path,
            get_path="path",
            get_color=[0, 100, 200],
            width_scale=10,
            width_min_pixels=3,
        )

        view_state = pdk.ViewState(
            latitude=df_points["lat"].mean(),
            longitude=df_points["lon"].mean(),
            zoom=14
        )

        route_map = pdk.Deck(
        layers=[point_layer, route_layer, text_layer],
        initial_view_state=view_state
    )

        st.pydeck_chart(route_map)


        st.subheader("Distances")
        st.dataframe(df_points[["name", "distance","unit"]])
