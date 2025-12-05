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

# Raise ValueErrors if environment variables are not set
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


col1, col2,col3  = st.columns(3)
with col1:
    minimizer = st.radio('Do you want to minimize:', ('duration','distance'), horizontal=True)
# Step 1 — Let user choose how to provide the image
with col2:
    choice = st.radio(
    "Choose how to provide an image:",
    ("Upload a file", "Take a picture"),
     horizontal=True)
with col3:
    road_mode = st.radio('Select a your drive mode', ('Car 🚗','Bike 🚴','Foot 👟'), horizontal=True)
if road_mode == 'Car 🚗':
    final_road_mode = "driving-car"
elif road_mode == 'Bike 🚴':
    final_road_mode = "cycling-regular"
elif road_mode == 'Foot 👟':
    final_road_mode = "foot-walking"

col4,col5,col6 = st.columns(3)
with col4:
    misc = st.checkbox("Check here to throw miscellaneous trash 🗑 on the road")
with col5:
    recycling = st.checkbox("Check here to throw recycling trash ♻️ on the road")
with col6:
    ress = st.checkbox("Check here to add good-looking items to a ressourcerie on the road")

with st.expander("Your selections"):
    st.write(f"- Miscellaneous trash: {'On the road' if misc else 'At home'}")
    st.write(f"- Recycling trash: {'On the road' if recycling else 'At home'}")
    st.write(f"- Good-looking items to ressourcerie: {'Added to the road' if ress else 'Not added to the road'}")

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
        #Ajuste la liste de prédiction en fonction des choix de l'user:
        print(f"from predict:{result_list= }")
        if not recycling:
            result_list =  [r for r in result_list if r.get("class") not in ["metal", "paper", "cardboard"]]
        if not misc:
            result_list =  [r for r in result_list if r.get("class") not in ["miscellaneous_trash"]]
        if ress:
            result_list.append({
                "class": "ressourcerie",
                "confidence": 0.9
            })
        print(f"after user input:{result_list= }")
        #To be added: ressourcerie
        # Mise à jour du conteneur avec le résultat
        status.success("Prediction done!")  # change le message
        for i, item in enumerate(result_list):
                    st.subheader(f"Prediction {i+1}")
                    st.write(item)

        progress_bar = st.progress(0)
        status_text = st.empty()
        def update_progress(fraction):
            progress_bar.progress(fraction)
            status_text.write(f"Processing… {int(fraction*100)}%")
        road_client = openrouteservice.Client(key=GEO_SERVICE_API_KEY)
        pick_up = {"lat": user_input[0],
                      "lon":user_input[1],
                      "trash_type":"User Start Point",
                      "distance": 0}
        drop_off_list = get_dropoff(road_client=road_client,
                                    result_list= result_list,
                                    starting_point = pick_up,
                                    prob_threshold = 0.05,
                                    profile=final_road_mode,
                                    minimizer=minimizer,
                                    keep_top_k=20,
                                    progress_callback=update_progress
                                    )


        all_routes = []
        all_paths = []
        all_points = []

        all_points.append({
                "lon": pick_up["lon"],
                "lat": pick_up["lat"],
                "name": "Start",
                "trash_type": "Start",
                "distance":0,
                "unit":" "
            })

        for drop_off in drop_off_list:
            # Calcul de route
            route = get_route(all_points[-1], drop_off, road_client)
            coords = route["features"][0]["geometry"]["coordinates"]

            # Stockage du chemin
            all_paths.append({"trash_type": drop_off["trash_type"], "path": coords})

            # Stockage des points (départ + arrivée)

            all_points.append({
                "lon": drop_off["lon"],
                "lat": drop_off["lat"],
                "trash_type": drop_off["trash_type"],
                "distance_m":drop_off["distance_m"],
                "duration_s":drop_off["duration_s"]
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
        get_text="trash_type",
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
        # Génération du texte pour chaque ligne
        for step_index, (idx, row) in enumerate(df_points.iloc[1:].iterrows(), start=1):
            minutes = row["duration_s"] / 60
            kilometers = row["distance_m"] / 1000

            step_text = (
                f"### Step {step_index}\n"
                f"**Destination:** Drop off your **{row['trash_type']}** waste\n"
                f"**Estimated time:** {minutes:.1f} minutes\n"
                f"**Distance:** {kilometers:.2f} km\n\n"
            )

            st.markdown(step_text)
