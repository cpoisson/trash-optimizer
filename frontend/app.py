import streamlit as st
from PIL import Image
from dummy_function import dummy_predict, dummy_get_drop_off
from dummy_function import DUMMY_START_POINT
import pydeck as pdk
import pandas as pd
import requests
from openrouteservice import convert
import openrouteservice
from key import GEO_KEY
from route_functions import get_route,get_dropoff

st.title("Trash-optimizer Front")

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
if img:
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
        response = requests.post("http://localhost:8000/predict", files=files)
        result_list = response.json()

        # Calcul de la prédiction
        # result_dict = dummy_predict()

        # Mise à jour du conteneur avec le résultat
        status.success("Prediction done!")  # change le message
        for i, item in enumerate(result_list):
                    st.subheader(f"Prediction {i+1}")
                    st.write(item)

        client = openrouteservice.Client(key=GEO_KEY)
        pick_up = DUMMY_START_POINT
        drop_off = get_dropoff(client=client)

        route = get_route(pick_up,drop_off,client)
        coords = route["features"][0]["geometry"]["coordinates"]

                # Conversion des coords en DataFrame
        df_path = pd.DataFrame({
            "path": [coords]
        })

        # Points de début/fin pour afficher des marqueurs
        df_points = pd.DataFrame([
    {"lon": pick_up["lon"], "lat": pick_up["lat"], "name": "Start"},
    {"lon": drop_off["lon"], "lat": drop_off["lat"], "name": "End"},
])

                # Layer : points
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            df_points,
            get_position='[lon, lat]',
            get_color='[200, 30, 0]',
            get_radius=40,
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
            layers=[point_layer, route_layer],
            initial_view_state=view_state
        )

        st.pydeck_chart(route_map)
