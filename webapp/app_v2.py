import hashlib
import os
from typing import Dict, List, Optional

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

from dummy_function import DUMMY_PROBABILITY_THRESHOLD
from route_functions import get_dropoff, get_route

# Load environment variables
load_dotenv()
GEO_SERVICE_API_KEY = os.getenv("GEO_SERVICE_API_KEY")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL")
UPLOAD_KEY = "upload_images"

# Validate environment variables early
if not GEO_SERVICE_API_KEY:
    raise ValueError("GEO_SERVICE_API_KEY is not set in environment variables.")
if not INFERENCE_SERVICE_URL:
    raise ValueError("INFERENCE_SERVICE_URL is not set in environment variables.")

# Session state init
if "images" not in st.session_state:
    st.session_state.images: List[Dict] = []
if "image_hashes" not in st.session_state:
    st.session_state.image_hashes = set()
if "skip_add_once" not in st.session_state:
    st.session_state.skip_add_once = False


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def trigger_rerun():
    """Trigger a rerun compatible with recent Streamlit."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def add_images(new_files: List[bytes], names: List[str]):
    """Store uploaded/captured images in session, skipping duplicates."""
    added = 0
    for content, name in zip(new_files, names):
        h = _hash_bytes(content)
        if h in st.session_state.image_hashes:
            continue
        st.session_state.images.append({"name": name, "content": content, "hash": h})
        st.session_state.image_hashes.add(h)
        added += 1
    if added == 0 and new_files:
        st.info("These images were already in the queue.")


def remove_image(index: int):
    img = st.session_state.images.pop(index)
    # Keep hash in set to avoid re-adding the same file on rerun


def fetch_predictions(img_bytes: bytes) -> Optional[List[Dict]]:
    """Call inference API for one image."""
    try:
        files = {"file": img_bytes}
        response = requests.post(f"{INFERENCE_SERVICE_URL}/predict", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.warning(f"Prediction failed for one image: {exc}")
        return None


def consolidate_classes(predictions: List[Dict], misc: bool, recycling: bool, ress: bool) -> List[Dict]:
    """Deduplicate by class, keep max confidence, apply user preferences."""
    class_conf: Dict[str, float] = {}
    for pred in predictions:
        cls = pred["class"]
        conf = pred["confidence"]
        if cls not in class_conf or conf > class_conf[cls]:
            class_conf[cls] = conf

    if not recycling:
        for cls in ["metal", "paper", "cardboard"]:
            class_conf.pop(cls, None)
    if not misc:
        class_conf.pop("miscellaneous_trash", None)
    if ress:
        class_conf["ressourcerie"] = max(class_conf.get("ressourcerie", 0), 0.9)
    return [{"class": cls, "confidence": conf} for cls, conf in class_conf.items()]


def display_predictions_ui(per_image_preds: List[Dict]):
    """Show per-image predictions as compact cards."""
    for item in per_image_preds:
        card = st.container(border=True)
        with card:
            cols = st.columns([1, 2])
            if thumb := item.get("content"):
                with cols[0]:
                    st.image(thumb, width=130)
            top1_conf = item["top1"]["confidence"]
            display_list = item["top3"] if top1_conf < 0.7 else [item["top1"]]
            with cols[1]:
                st.markdown(f"**Prediction:** {display_list[0]['class']}  ({display_list[0]['confidence']:.2f})")
                st.caption("Where should I throw this? Follow the route steps below.")
                if top1_conf < 0.7 and len(display_list) > 1:
                    st.caption("Low confidence — other possibilities:")
                    st.write("\n".join([f"- {p['class']} ({p['confidence']:.2f})" for p in display_list[1:]]))


def get_category_color(category: str) -> List[int]:
    palette = {
        "miscellaneous_trash": [200, 80, 60],
        "metal": [80, 120, 200],
        "paper": [120, 200, 240],
        "cardboard": [180, 140, 80],
        "plastic": [240, 180, 80],
        "glass": [100, 200, 180],
        "textile_trash": [200, 120, 200],
        "vegetation": [120, 200, 120],
        "ressourcerie": [160, 120, 240],
    }
    return palette.get(category, [0, 100, 200])


def render_step_cards(df_points: pd.DataFrame):
    """Render route steps as small cards."""
    for step_index, (_, row) in enumerate(df_points.iloc[1:].iterrows(), start=1):
        card = st.container(border=True)
        with card:
            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown(f"**Step {step_index}: Drop off {row['trash_type']}**")
                st.caption("Follow the map to reach this stop.")
            with cols[1]:
                st.markdown(f"Distance: **{row['distance_m'] / 1000:.2f} km**")
                st.markdown(f"Duration: **{row['duration_s'] / 60:.1f} min**")


st.set_page_config(page_title="Trash Optimizer", layout="wide", page_icon="🗑️")
st.title("Trash Optimizer")
st.caption("Simple route planning based on your photos. Steps: Address → Images → Route.")

st.markdown("### 1) Start address")
st.markdown("Enter where your route should start. We'll locate it for you.")
address = st.text_input("Your address", value="Nantes", placeholder="e.g., 123 Main St, City")
user_input = None
if address:
    geolocator = Nominatim(user_agent="streamlit_app_v2")
    try:
        location = geolocator.geocode(address)
        if location:
            st.success("Great, we found your starting point.")
            user_input = (location.latitude, location.longitude)
        else:
            st.error("We couldn't find that address. Try a nearby landmark.")
    except Exception as exc:
        st.error("Geocoding failed. Please try again in a moment.")

st.markdown("### 2) Options")
col1, col2, col3 = st.columns(3)
with col1:
    minimizer = st.radio("Optimize for", ("duration", "distance"), horizontal=True)
with col2:
    choice = st.radio("Provide images", ("Upload files", "Take a picture"), horizontal=True)
with col3:
    road_mode = st.radio("Travel mode", ("Car 🚗", "Bike 🚴", "Foot 👟"), horizontal=True)
if road_mode == "Car 🚗":
    final_road_mode = "driving-car"
elif road_mode == "Bike 🚴":
    final_road_mode = "cycling-regular"
else:
    final_road_mode = "foot-walking"

col4, col5, col6 = st.columns(3)
with col4:
    misc = st.checkbox("Miscellaneous trash 🗑 on the road", value=True)
with col5:
    recycling = st.checkbox("Recycling trash ♻️ on the road", value=True)
with col6:
    ress = st.checkbox("Add good-looking items to a ressourcerie")
st.caption(f"Misc: {'on the road' if misc else 'at home'} | Recycling: {'on the road' if recycling else 'at home'} | Ressourcerie: {'added' if ress else 'not added'}")

# Image inputs
st.markdown("### 3) Images to analyze")
col_upload, col_camera = st.columns([2, 1])
if choice == "Upload files":
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=UPLOAD_KEY,
        )
        if uploaded_files and not st.session_state.get("skip_add_once", False):
            add_images([f.read() for f in uploaded_files], [f.name for f in uploaded_files])
        st.session_state.skip_add_once = False
    with col_camera:
        st.caption("Or take a picture")
        camera_image = st.camera_input("Camera shot")
        if camera_image:
            add_images([camera_image.getvalue()], ["camera_capture"])
else:
    with col_camera:
        camera_image = st.camera_input("Camera shot")
        if camera_image:
            add_images([camera_image.getvalue()], ["camera_capture"])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=UPLOAD_KEY,
        )
        if uploaded_files and not st.session_state.get("skip_add_once", False):
            add_images([f.read() for f in uploaded_files], [f.name for f in uploaded_files])
        st.session_state.skip_add_once = False

# Show current queue
if st.session_state.images:
    st.write("### Selected images")
    st.caption(f"{len(st.session_state.images)} image(s) queued")
    if st.button("Clear all"):
        st.session_state.images = []
        st.session_state.image_hashes = set()
        st.session_state.skip_add_once = True
        trigger_rerun()
    grid_cols = st.columns(3)
    for idx, item in enumerate(st.session_state.images):
        col = grid_cols[idx % 3]
        with col:
            st.write(item["name"])
            st.image(item["content"], width=180)
            if st.button("Remove", key=f"remove_{idx}"):
                remove_image(idx)
                trigger_rerun()
else:
    st.info("Add at least one image (upload or camera) to continue.")

st.divider()

keep_top_k = 20
prob_threshold = float(DUMMY_PROBABILITY_THRESHOLD)

run_pred = st.button("Run predictions and build route", type="primary", disabled=not st.session_state.images or not user_input)

if run_pred:
    per_image_preds = []
    consolidated_inputs = []
    failed_images = []

    progress_bar = st.progress(0)
    for idx, img in enumerate(st.session_state.images):
        progress_bar.progress((idx) / len(st.session_state.images))
        preds = fetch_predictions(img["content"])
        if preds is None:
            failed_images.append(img["name"])
            continue

        # Sort predictions by confidence desc
        sorted_preds = sorted(preds, key=lambda x: x.get("confidence", 0), reverse=True)
        top1 = sorted_preds[0] if sorted_preds else {"class": "unknown", "confidence": 0.0}
        top3 = sorted_preds[:3]
        per_image_preds.append({
            "name": img["name"],
            "top1": top1,
            "top3": top3,
            "content": img["content"],
        })
        consolidated_inputs.append(top1)

    progress_bar.progress(1.0)

    if failed_images:
        st.warning(f"Predictions failed for: {', '.join(failed_images)}")

    if not consolidated_inputs:
        st.error("No successful predictions to process.")
    else:
        st.write("### Predictions per image")
        display_predictions_ui(per_image_preds)

        consolidated = consolidate_classes(consolidated_inputs, misc=misc, recycling=recycling, ress=ress)

        if not consolidated:
            st.error("All predicted classes were filtered out. Adjust your selections or try different images.")
        else:
            st.success(f"Classes sent to routing: {[c['class'] for c in consolidated]}")

            # Route computation
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(fraction):
                progress_bar.progress(min(fraction, 1.0))
                status_text.write(f"Processing… {int(fraction*100)}%")

            road_client = None
            import openrouteservice
            road_client = openrouteservice.Client(key=GEO_SERVICE_API_KEY)
            pick_up = {
                "lat": user_input[0],
                "lon": user_input[1],
                "trash_type": "User Start Point",
                "distance": 0,
            }

            drop_off_list = get_dropoff(
                road_client=road_client,
                result_list=consolidated,
                starting_point=pick_up,
                prob_threshold=prob_threshold,
                profile=final_road_mode,
                minimizer=minimizer,
                keep_top_k=keep_top_k,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0)
            status_text.write("Routing complete.")

            if not drop_off_list:
                st.error("No drop-off points found for the selected classes.")
            else:
                all_paths = []
                all_points = []

                all_points.append({
                    "lon": pick_up["lon"],
                    "lat": pick_up["lat"],
                    "name": "Start",
                    "trash_type": "Start",
                    "distance": 0,
                    "unit": " ",
                })

                for drop_off in drop_off_list:
                    route = get_route(all_points[-1], drop_off, road_client, profile=final_road_mode)
                    coords = route["features"][0]["geometry"]["coordinates"]
                    all_paths.append({"trash_type": drop_off["trash_type"], "path": coords})
                    all_points.append({
                        "lon": drop_off["lon"],
                        "lat": drop_off["lat"],
                        "trash_type": drop_off["trash_type"],
                        "distance_m": drop_off["distance_m"],
                        "duration_s": drop_off["duration_s"],
                    })

                df_path = pd.DataFrame(all_paths)
                df_points = pd.DataFrame(all_points)

                point_layer = pdk.Layer(
                    "ScatterplotLayer",
                    df_points,
                    get_position="[lon, lat]",
                    get_color="[200, 30, 0]",
                    get_radius=20,
                )
                text_layer = pdk.Layer(
                    "TextLayer",
                    df_points,
                    pickable=True,
                    get_position="[lon, lat]",
                    get_text="trash_type",
                    get_color=[255, 255, 0],
                    get_size=20,
                    get_alignment_baseline="'bottom'",
                )
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
                    zoom=14,
                )

                route_map = pdk.Deck(
                    layers=[point_layer, route_layer, text_layer],
                    initial_view_state=view_state,
                )

                st.pydeck_chart(route_map)

                st.subheader("Steps")
                if len(df_points) > 1:
                    steps_rows = []
                    for step_index, (_, row) in enumerate(df_points.iloc[1:].iterrows(), start=1):
                        steps_rows.append({
                            "Step": step_index,
                            "Type": row["trash_type"],
                            "Distance (km)": round(row["distance_m"] / 1000, 2),
                            "Duration (min)": round(row["duration_s"] / 60, 1),
                        })
                    steps_df = pd.DataFrame(steps_rows)
                    st.table(steps_df)
