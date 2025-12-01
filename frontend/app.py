import streamlit as st
from PIL import Image
from dummy_function import dummy_predict, dummy_get_drop_off, DUMMY_START_POINT
import pydeck as pdk
import pandas as pd
import requests

st.title("Trash-optimizer Front")

# Step 1 — Let user choose how to provide the image
choice = st.radio(
    "Choose how to provide an image:",
    ("Upload a file", "Take a picture")
)

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
        files = {"file": uploaded_file.getvalue()}  # envoie l'image brute
        # Si ton API attend plutôt un form-data multipart
        result_list = requests.post("http://localhost:8000/predict", files=files)

        # Calcul de la prédiction
        # result_dict = dummy_predict()

        # Mise à jour du conteneur avec le résultat
        status.success("Prediction done!")  # change le message
        for i, item in enumerate(result_list):
                    st.subheader(f"Prediction {i+1}")
                    st.write(item)

        drop_off = dummy_get_drop_off()
        pick_up = DUMMY_START_POINT

        # Points dataframe
        df_points = pd.DataFrame([
            {"lat": pick_up["lat"], "lon": pick_up["lon"], "name": "Pick Up"},
            {"lat": drop_off["lat"], "lon": drop_off["lon"], "name": "Drop Off"},
        ])

        # Line dataframe
        df_line = pd.DataFrame([{
            "source": [pick_up["lon"], pick_up["lat"]],
            "target": [drop_off["lon"], drop_off["lat"]],
        }])

        point_layer = pdk.Layer(
            "ScatterplotLayer",
            df_points,
            get_position='[lon, lat]',
            get_radius=30,
            get_color=[200, 30, 0],
            pickable=True
        )

        line_layer = pdk.Layer(
            "LineLayer",
            df_line,
            get_source_position='source',
            get_target_position='target',
            get_color=[0, 100, 200],
            get_width=4
        )

        view = pdk.ViewState(
            latitude=(pick_up["lat"] + drop_off["lat"]) / 2,
            longitude=(pick_up["lon"] + drop_off["lon"]) / 2,
            zoom=14
        )

        r = pdk.Deck(
            layers=[point_layer, line_layer],
            initial_view_state=view,
            tooltip={"text": "{name}"}
        )

        st.pydeck_chart(r)
