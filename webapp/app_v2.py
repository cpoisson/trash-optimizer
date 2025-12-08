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
st.session_state.setdefault("images", [])
st.session_state.setdefault("image_hashes", set())


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def trigger_rerun():
    """Trigger a rerun compatible with recent Streamlit."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def add_images(new_files: List[bytes], names: List[str]):
    """Store uploaded/captured images, skipping duplicates by hash."""
    for content, name in zip(new_files, names):
        h = _hash_bytes(content)
        if h in st.session_state.image_hashes:
            continue
        st.session_state.images.append({"name": name, "content": content, "hash": h})
        st.session_state.image_hashes.add(h)


def remove_image(index: int):
    """Remove image at index."""
    st.session_state.images.pop(index)


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
                st.markdown(f"**{display_list[0]['confidence']*100:.0f}%** {humanize_label(display_list[0]['class'])}")
                if top1_conf < 0.7 and len(display_list) > 1:
                    st.caption("Low confidence — other possibilities:")
                    st.write("\n".join([f"- {humanize_label(p['class'])} ({p['confidence']*100:.0f}%)" for p in display_list[1:]]))


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


def humanize_label(label: str) -> str:
    """Make class names user-friendly for display."""
    return label.replace("_", " ").title()


def render_step_cards(df_points: pd.DataFrame):
    """Render route steps as small cards."""
    for step_index, (_, row) in enumerate(df_points.iloc[1:].iterrows(), start=1):
        card = st.container(border=True)
        with card:
            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown(f"**Step {step_index}: Drop off {humanize_label(row['trash_type'])}**")
                st.caption("Follow the map to reach this stop.")
            with cols[1]:
                st.markdown(f"Distance: **{row['distance_m'] / 1000:.2f} km**")
                st.markdown(f"Duration: **{row['duration_s'] / 60:.1f} min**")


def compute_view_state(df_points: pd.DataFrame):
    """Compute a view state that keeps all points visible without over-zooming."""
    if df_points.empty:
        return pdk.ViewState(latitude=0, longitude=0, zoom=12)
    min_lat, max_lat = df_points["lat"].min(), df_points["lat"].max()
    min_lon, max_lon = df_points["lon"].min(), df_points["lon"].max()
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    lat_span = max(max_lat - min_lat, 1e-4) * 1.3
    lon_span = max(max_lon - min_lon, 1e-4) * 1.3
    span = max(lat_span, lon_span)
    zoom = max(8, min(15, 12 - span * 60))
    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)


# --------------------------------------
# Layout and user flow
# --------------------------------------
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

# Image inputs
st.markdown("### 3) Images to analyze")
col_upload, col_camera = st.columns([2, 1])
with col_upload:
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key=UPLOAD_KEY,
    )
    if uploaded_files:
        add_images([f.read() for f in uploaded_files], [f.name for f in uploaded_files])
with col_camera:
    st.caption("Or take a picture")
    camera_image = st.camera_input("Camera shot")
    if camera_image:
        add_images([camera_image.getvalue()], ["camera_capture"])

# Show current queue
if st.session_state.images:
    st.write("### Selected images")
    grid_cols = st.columns(3)
    for idx, item in enumerate(st.session_state.images):
        col = grid_cols[idx % 3]
        with col:
            st.markdown(f"**{item['name']}**")
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

    for img in st.session_state.images:
        preds = fetch_predictions(img["content"])
        if preds is None:
            failed_images.append(img["name"])
            continue

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

            road_client = None
            import openrouteservice
            road_client = openrouteservice.Client(key=GEO_SERVICE_API_KEY)
            pick_up = {
                "lat": user_input[0],
                "lon": user_input[1],
                "trash_type": "Start",
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
                progress_callback=None,
            )

            if not drop_off_list:
                st.error("No drop-off points found for the selected classes.")
            else:
                all_paths = []
                all_points = []

                all_points.append({
                    "lon": pick_up["lon"],
                    "lat": pick_up["lat"],
                    "trash_type": "Start",
                    "label": "Start",
                    "color": [255, 0, 0],   # start in red
                    "distance": 0,
                    "unit": " ",
                })

                for idx_step, drop_off in enumerate(drop_off_list, start=1):
                    route = get_route(all_points[-1], drop_off, road_client, profile=final_road_mode)
                    coords = route["features"][0]["geometry"]["coordinates"]
                    # Single color for all segments
                    path_color = [0, 100, 200]
                    all_paths.append({"trash_type": drop_off["trash_type"], "path": coords, "color": path_color})
                    all_points.append({
                        "lon": drop_off["lon"],
                        "lat": drop_off["lat"],
                        "trash_type": drop_off["trash_type"],
                        "label": f"{idx_step}",
                        "color": [255, 215, 0],  # yellow for stops
                        "distance_m": drop_off["distance_m"],
                        "duration_s": drop_off["duration_s"],
                    })

                df_path = pd.DataFrame(all_paths)
                df_points = pd.DataFrame(all_points)

                point_layer = pdk.Layer(
                    "ScatterplotLayer",
                    df_points,
                    get_position="[lon, lat]",
                    get_color="color",
                    get_radius=40,
                )
                text_layer = pdk.Layer(
                    "TextLayer",
                    df_points,
                    pickable=True,
                    get_position="[lon, lat]",
                    get_text="label",
                    get_color=[0, 0, 0],
                    get_size=16,
                    get_alignment_baseline="'middle'",
                    get_pixel_offset=[0, -10],
                )
                route_layer = pdk.Layer(
                    "PathLayer",
                    df_path,
                    get_path="path",
                    get_color="color",
                    width_scale=10,
                    width_min_pixels=3,
                )

                view_state = compute_view_state(df_points)

                route_map = pdk.Deck(
                    layers=[point_layer, route_layer, text_layer],
                    initial_view_state=view_state,
                )

                st.pydeck_chart(route_map)

                st.subheader("Steps")
                if len(df_points) > 1:
                    render_step_cards(df_points)
