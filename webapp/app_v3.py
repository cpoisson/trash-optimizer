import streamlit as st
from PIL import Image, ImageDraw
import hashlib
from typing import cast, Any
import requests
import pandas as pd
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import openrouteservice
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import os
import io
import json
import logging
from typing import Dict, List, Tuple, Optional
from route_functions import get_route_matrix, get_route, get_dropoff
from query.query import get_loc

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/trash_optimizer_v3.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Starting Trash Optimizer V3 Application")
logger.info("=" * 80)

# Load environment variables
load_dotenv()
GEO_SERVICE_API_KEY = os.getenv("GEO_SERVICE_API_KEY")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")

# Fallback location (Nantes) so the map never renders empty
DEFAULT_LOCATION = {"lat": 47.218371, "lon": -1.553621}

logger.info(f"Inference Service URL: {INFERENCE_SERVICE_URL}")
logger.info(f"GEO_SERVICE_API_KEY configured: {'Yes' if GEO_SERVICE_API_KEY else 'No'}")

# Validate environment variables
if not GEO_SERVICE_API_KEY:
    logger.error("GEO_SERVICE_API_KEY is not set. Please configure your environment.")
    st.error("GEO_SERVICE_API_KEY is not set. Please configure your environment.")
    st.stop()

# Initialize OpenRouteService client
logger.debug("Initializing OpenRouteService client")
ors_client = openrouteservice.Client(key=GEO_SERVICE_API_KEY)
logger.debug("OpenRouteService client initialized successfully")

# Page config
st.set_page_config(
    page_title="Trash Optimizer",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "screen" not in st.session_state:
    st.session_state.screen = 0  # 0: Welcome, 1: Location, 2: Items, 3: Map

if "user_location" not in st.session_state:
    st.session_state.user_location = None

if "trash_items" not in st.session_state:
    st.session_state.trash_items = []  # List of {image, category, confidence}

if "drop_off_locations" not in st.session_state:
    st.session_state.drop_off_locations = None

if "transport_mode" not in st.session_state:
    st.session_state.transport_mode = "driving-car"

if "cached_route" not in st.session_state:
    st.session_state.cached_route = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def classify_trash_image(image: Image.Image) -> Dict:
    """
    Send image to inference service for classification.
    Returns dict with top prediction: {"class": str, "confidence": float}
    """
    try:
        logger.debug("Starting image classification")

        # Ensure RGB to avoid Pillow RGBA/JPEG errors
        if image.mode != "RGB":
            image = image.convert("RGB")
            logger.debug(f"Image converted to RGB from {image.mode}")

        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        logger.debug(f"Image converted to JPEG bytes, size: {len(img_bytes.getvalue())} bytes")

        # Send to inference API
        logger.info(f"Sending classification request to {INFERENCE_SERVICE_URL}/predict")
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(
            f"{INFERENCE_SERVICE_URL}/predict",
            files=files,
            timeout=30
        )

        logger.debug(f"Classification API response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logger.debug(f"Classification result: {result}")

            predictions = []
            if isinstance(result, list):
                predictions = result
            elif isinstance(result, dict):
                predictions = result.get("predictions") or result.get("results") or []

            if predictions:
                # Ensure highest confidence first
                sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)
                top_pred = sorted_preds[0]
                class_name = top_pred.get("class", "unknown")
                confidence = float(top_pred.get("confidence", 0.0))

                logger.info(f"Top prediction: {class_name} (confidence: {confidence:.2%})")
                return {
                    "class": class_name,
                    "confidence": confidence
                }
            else:
                logger.warning("No predictions in classification response")
        else:
            logger.error(f"Classification API returned status {response.status_code}: {response.text}")

        return {"class": "unknown", "confidence": 0.0}
    except requests.exceptions.Timeout:
        logger.error("Classification request timed out (30s)")
        st.error("Classification request timed out. Please try again.")
        return {"class": "unknown", "confidence": 0.0}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to inference service at {INFERENCE_SERVICE_URL}: {e}")
        st.error(f"Cannot connect to inference service. Is it running at {INFERENCE_SERVICE_URL}?")
        return {"class": "unknown", "confidence": 0.0}
    except Exception as e:
        logger.exception(f"Classification error: {type(e).__name__}: {e}")
        st.error(f"Classification error: {e}")
        return {"class": "unknown", "confidence": 0.0}


def geocode_address(address: str) -> Optional[Dict]:
    """Convert address to coordinates."""
    try:
        logger.info(f"Geocoding address: {address}")
        geolocator = Nominatim(user_agent="trash_optimizer_v3")
        location = geolocator.geocode(address)

        if location:
            # cast for type-checkers; geocode returns a Location
            location = cast(Any, location)
            logger.info(f"Address '{address}' geocoded successfully: lat={location.latitude}, lon={location.longitude}")
            return {"lat": location.latitude, "lon": location.longitude}
        else:
            logger.warning(f"Could not geocode address: {address} (not found)")
            st.error(f"Address not found: {address}")
            return None
    except Exception as e:
        logger.exception(f"Geolocation error for '{address}': {type(e).__name__}: {e}")
        # Show user-friendly error message
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            st.error("⚠️ Cannot connect to location service. Please check your internet connection and try again.")
        else:
            st.error(f"⚠️ Unable to find location. Please check the address and try again.")
        return None


def get_drop_off_points(trash_categories: List[str], user_location: Dict) -> pd.DataFrame:
    """
    Fetch drop-off points for given trash categories from BigQuery.
    """
    try:
        logger.info(f"Fetching drop-off points for categories: {trash_categories}")
        logger.debug(f"User location: {user_location}")

        if not trash_categories:
            logger.warning("No trash categories provided for drop-off point search")
            return pd.DataFrame()

        result = get_loc(trash_categories)
        logger.info(f"Successfully fetched {len(result)} drop-off points from BigQuery")

        if len(result) > 0:
            logger.debug(f"Drop-off point categories: {result['Trash_class'].unique().tolist()}")

        return result
    except Exception as e:
        logger.exception(f"Error fetching drop-off points: {type(e).__name__}: {e}")
        st.warning(f"Could not fetch drop-off points: {e}")
        # Return empty dataframe if BigQuery fails
        return pd.DataFrame()


def calculate_route(
    start: Dict,
    trash_categories: List[str],
    profile: str = "driving-car",
    progress_callback=None
) -> Tuple[Optional[List], Optional[List]]:
    """
    Calculate optimized route using get_dropoff to find best drop-off points.
    Returns: (route_segments, ordered_drop_offs)
    Each segment contains: distance, duration (no detailed turn-by-turn for performance)
    progress_callback: Optional function to report progress (0.0 to 1.0)
    """
    try:
        logger.info(f"Calculating route with profile: {profile}")
        logger.debug(f"Start point: {start}")
        logger.debug(f"Trash categories: {trash_categories}")

        # Create result_list format expected by get_dropoff (with confidence)
        result_list = [{"class": cat, "confidence": 1.0} for cat in trash_categories]

        # Use get_dropoff to find optimized drop-off points
        logger.info("Finding optimized drop-off points using get_dropoff")
        if progress_callback:
            progress_callback(0.2, "🔍 Searching collection points database...")

        optimized_dropoffs = get_dropoff(
            road_client=ors_client,
            result_list=result_list,
            starting_point=start,
            prob_threshold=0.1,  # Accept all since we're using confidence=1.0
            profile=profile,
            minimizer="distance",
            keep_top_k=10,  # Reduced from 15 to 10 for faster performance
            progress_callback=None
        )

        if progress_callback:
            progress_callback(0.5, "📍 Found collection points! Calculating routes...")

        if not optimized_dropoffs:
            logger.warning("No drop-off points found")
            return None, None

        logger.info(f"Found {len(optimized_dropoffs)} optimized drop-off points")

        # Build waypoints list with pre-calculated distances and fetch real route geometry
        waypoints = []
        route_segments = []

        # Track previous point for route geometry
        prev_point = start

        for idx, drop_off in enumerate(optimized_dropoffs):
            waypoint = {
                "coords": [drop_off["lon"], drop_off["lat"]],
                "lat": drop_off["lat"],
                "lon": drop_off["lon"],
                "name": f"Drop-off for {drop_off['trash_type']}",
                "trash_class": drop_off["trash_type"],
                "distance_m": drop_off.get("distance_m", 0),
                "duration_s": drop_off.get("duration_s", 0),
                "address": drop_off.get("address", "Address unavailable"),
                "location_name": drop_off.get("name", "")
            }
            waypoints.append(waypoint)

            # Fetch real route geometry for this segment
            route_geojson = None
            curr_point = {"lat": drop_off["lat"], "lon": drop_off["lon"]}
            try:
                logger.debug(f"Fetching route geometry for segment {idx+1}")

                # Report progress for each route segment
                if progress_callback:
                    segment_progress = 0.5 + (0.4 * (idx + 1) / len(optimized_dropoffs))
                    progress_callback(segment_progress, f"🛣️ Getting directions to stop {idx+1}/{len(optimized_dropoffs)}...")

                route_geojson = get_route(prev_point, curr_point, ors_client, profile)
                logger.debug(f"Route geometry fetched for segment {idx+1}")
            except Exception as route_err:
                logger.warning(f"Could not fetch route geometry for segment {idx+1}: {route_err}")

            # Create segment with real route geometry
            route_segments.append({
                "from": "Your Location" if idx == 0 else f"Drop-off for {optimized_dropoffs[idx-1]['trash_type']}",
                "to": waypoint["name"],
                "distance": drop_off.get("distance_m", 0),
                "duration": drop_off.get("duration_s", 0),
                "geojson": route_geojson
            })

            prev_point = curr_point

        logger.info(f"Built {len(waypoints)} waypoints and {len(route_segments)} segments with route geometry")
        return route_segments, waypoints

    except Exception as e:
        logger.exception(f"Route calculation error: {type(e).__name__}: {e}")
        st.error(f"Route calculation error: {e}")
        return None, None
# ============================================================================
# SCREEN 0: WELCOME SCREEN
# ============================================================================

def screen_welcome():
    """Welcome screen with engaging call-to-action."""
    logger.debug("Rendering welcome screen")
    st.image("assets/logo-lewagon.png", width=300)

    st.markdown("## ♻️ Let's Trash Some Stuff!")
    st.markdown(
        """
        ### Give me your trash, I'll lead you where you can trash it.

        No hassle. No judgment. Just smart routing to your local recycling & disposal spots.

        **Ready to make a difference?** Let's get started!
        """
    )

    # Navigation buttons
    col_next, col_empty = st.columns([1, 3])
    with col_next:
        if st.button("🚀 Get Started", key="welcome_next", use_container_width=True):
            logger.info("User clicked 'Get Started' - transitioning to location screen")
            st.session_state.screen = 1
            st.rerun()



# ============================================================================
# SCREEN 1: LOCATION INPUT
# ============================================================================

def screen_location():
    """Location input screen with transport mode."""
    logger.debug("Rendering location input screen")

    st.markdown("## 📍 Where Do You Live?")
    st.markdown("*We won't keep your location, but we need it to guide you to your local recycling places.*")

    # Address input
    address = st.text_input(
        "Enter your address:",
        value="Nantes, France",
        placeholder="City, Country or Full Address"
    )

    # Transport Option
    transport_option = st.radio(
        "Select your transport mode:",
        ["🚗 Driving", "🚴 Cycling", "👟 Walking"],
        horizontal=True,
        index=0
    )
    mode_map = {
        "🚗 Driving": "driving-car",
        "🚴 Cycling": "cycling-regular",
        "👟 Walking": "foot-walking"
    }
    st.session_state.transport_mode = mode_map[transport_option]
    logger.info(f"Transport mode selected: {st.session_state.transport_mode}")

    # Geocode and show location on map
    if address:
        location = geocode_address(address)
        if location:
            st.session_state.user_location = location
            logger.info(f"Location saved to session: {location}")
            st.success(f"✅ Location found: {address}")
        else:
            logger.warning(f"Failed to geocode address: {address}")

    # Always show a map (fallback to Nantes if nothing yet) - smaller size for mobile responsiveness
    map_location = st.session_state.user_location or DEFAULT_LOCATION

    # Create Folium map for location preview
    location_map = folium.Map(
        location=[map_location["lat"], map_location["lon"]],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    # Add marker for the location
    folium.Marker(
        location=[map_location["lat"], map_location["lon"]],
        popup="<b>📍 Your Location</b>",
        tooltip="Your location",
        icon=folium.Icon(color='blue', icon='home', prefix='fa')
    ).add_to(location_map)

    # Render map
    st_folium(location_map, width=None, height=300, use_container_width=True, returned_objects=[])


    # Navigation
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Back", key="location_back", use_container_width=True):
            logger.info("User clicked back from location screen")
            st.session_state.screen = 0
            st.rerun()

    with col_next:
        if st.session_state.user_location:
            if st.button("Next ➡️", key="location_next", use_container_width=True):
                logger.info("User confirmed location - transitioning to trash items screen")
                st.session_state.screen = 2
                st.rerun()
        else:
            st.button("Next ➡️", key="location_next_disabled", disabled=True, use_container_width=True)


# ============================================================================
# SCREEN 2: TRASH ITEM CLASSIFICATION
# ============================================================================

def get_category_icon(category: str) -> str:
    """Return emoji icon for category."""
    icons = {
        "food_organics": "🍎",
        "cardboard": "📦",
        "glass": "🍾",
        "metal": "🔩",
        "paper": "📄",
        "plastic": "♻️",
        "textile_trash": "👕",
        "vegetation": "🌿",
        "miscellaneous_trash": "🗑️",
        "mirror": "🪞",
        "car_battery": "🔋",
        "neon": "💡",
        "pharmacy": "💊",
        "tire": "🛞",
        "printer_cartridge": "🖨️",
        "wood": "🪵",
        "light_bulb": "💡",
        "battery": "🔋"
    }
    return icons.get(category, "📦")


def screen_trash_items():
    """Trash item classification screen - minimal design with auto-add."""
    logger.debug("Rendering trash items classification screen")

    st.markdown("## 🗑️ What Are We Trashing Today?")

    # Category list for consistency
    categories = [
        "food_organics", "cardboard", "glass", "metal", "paper",
        "plastic", "textile_trash", "vegetation", "miscellaneous_trash",
        "mirror", "car_battery", "neon", "pharmacy", "tire",
        "printer_cartridge", "wood", "light_bulb", "battery"
    ]

    # Add New Item section - collapsible expander at top
    if len(st.session_state.trash_items) < 4:
        with st.expander("➕ Add New Item", expanded=False):
            # Input method selector
            input_method = st.radio(
                "How would you like to add an item?",
                ["Upload Photo", "Take Picture"],
                horizontal=True,
                key="input_method_radio"
            )

            image = None
            if input_method == "Upload Photo":
                logger.debug("User selected 'Upload Photo' input method")
                uploaded_file = st.file_uploader(
                    "Choose an image",
                    type=["jpg", "jpeg", "png"],
                    key="trash_upload"
                )
                if uploaded_file:
                    logger.debug(f"File uploaded: {uploaded_file.name}")
                    image = Image.open(uploaded_file)

            else:  # Take Picture
                logger.debug("User selected 'Take Picture' input method")
                camera_image = st.camera_input("Take a photo of your trash")
                if camera_image:
                    logger.debug("Camera image captured")
                    image = Image.open(camera_image)

            # Auto-classify and auto-add
            if image:
                # Normalize to RGB
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Cache the prediction per image
                img_hash = hashlib.md5(image.tobytes()).hexdigest()

                # Check if this image was already added
                if img_hash not in [hashlib.md5(item["image"].tobytes()).hexdigest() for item in st.session_state.trash_items]:
                    if st.session_state.get("_last_img_hash") != img_hash:
                        with st.spinner("Analyzing and adding..."):
                            prediction = classify_trash_image(image)

                            # Auto-add to list
                            st.session_state.trash_items.append({
                                "image": image,
                                "category": prediction["class"] if prediction["class"] in categories else "miscellaneous_trash",
                                "confidence": float(prediction.get("confidence", 0.0))
                            })
                            logger.info(f"Auto-added trash item: {prediction['class']} (Total: {len(st.session_state.trash_items)}/4)")
                            st.session_state._last_img_hash = img_hash
                            st.rerun()
                else:
                    st.info("✅ This item has already been added!")

    else:
        st.info("✅ Maximum items (4) reached. Remove an item to add more.")

    # Items list - main focus of the screen (compact design with thumbnails)
    st.markdown("---")

    if st.session_state.trash_items:
        st.subheader(f"Items to Drop Off ({len(st.session_state.trash_items)}/4)")

        for idx, item in enumerate(st.session_state.trash_items):
            # Compact row for each item
            col1, col2, col3, col4 = st.columns([0.8, 2.5, 2, 0.7])

            with col1:
                # Small thumbnail
                st.image(item["image"], width=60)

            with col2:
                # Category with icon
                icon = get_category_icon(item["category"])
                st.markdown(f"**{icon} {item['category'].replace('_', ' ').title()}**")

                # Confidence badge (compact)
                conf_pct = item['confidence'] * 100
                if conf_pct >= 70:
                    st.caption(f"✅ {conf_pct:.0f}%")
                elif conf_pct >= 50:
                    st.caption(f"⚠️ {conf_pct:.0f}%")
                else:
                    st.caption(f"❓ {conf_pct:.0f}%")

            with col3:
                # Edit category dropdown (always available)
                current_idx = categories.index(item["category"]) if item["category"] in categories else 0

                def format_with_icon(cat):
                    return f"{get_category_icon(cat)} {cat.replace('_', ' ').title()}"

                new_category = st.selectbox(
                    "Category",
                    options=categories,
                    index=current_idx,
                    format_func=format_with_icon,
                    key=f"edit_cat_{idx}",
                    label_visibility="collapsed"
                )
                if new_category != item["category"]:
                    item["category"] = new_category
                    logger.info(f"User edited item {idx} category to: {new_category}")

            with col4:
                # Remove button
                if st.button("🗑️", key=f"remove_{idx}", use_container_width=True, help="Remove item"):
                    logger.info(f"User removed trash item {idx}: {item['category']}")
                    st.session_state.trash_items.pop(idx)
                    st.rerun()

            st.markdown("---")
    else:
        st.info("👆 Click **Add New Item** above to get started!")

    # Navigation at bottom
    st.markdown("")
    col_back, col_next = st.columns([1, 1])

    with col_back:
        if st.button("⬅️ Back", key="items_back", use_container_width=True):
            logger.info("User clicked back from items screen")
            st.session_state.screen = 1
            st.rerun()

    with col_next:
        if st.session_state.trash_items:
            if st.button("Next ➡️ Generate Route", key="items_next", use_container_width=True, type="primary"):
                logger.info(f"User proceeding to route generation with {len(st.session_state.trash_items)} items")
                st.session_state.screen = 3
                st.rerun()
        else:
            st.button("Next ➡️ Generate Route", key="items_next_disabled", disabled=True, use_container_width=True)


# ============================================================================
# SCREEN 3: ROUTE MAP
# ============================================================================

def screen_map():
    """Final map screen showing route to drop-off points."""
    logger.debug("Rendering route map screen")

    # Compact header with transport mode selector
    col_title, col_transport = st.columns([3, 1])
    with col_title:
        st.markdown("## 🗺️ Your Trash Route")
    with col_transport:
        mode_map_reverse = {
            "driving-car": "🚗 Driving",
            "cycling-regular": "🚴 Cycling",
            "foot-walking": "👟 Walking"
        }
        mode_map = {
            "🚗 Driving": "driving-car",
            "🚴 Cycling": "cycling-regular",
            "👟 Walking": "foot-walking"
        }
        transport_option = st.selectbox(
            "Transport",
            ["🚗 Driving", "🚴 Cycling", "👟 Walking"],
            index=list(mode_map.values()).index(st.session_state.transport_mode),
            key="map_transport_selector"
        )
        selected_mode = mode_map[transport_option]

        # If transport mode changed, update and recalculate
        if selected_mode != st.session_state.transport_mode:
            logger.info(f"Transport mode changed from {st.session_state.transport_mode} to {selected_mode}")
            st.session_state.transport_mode = selected_mode
            st.rerun()

    # Get trash categories from items
    trash_categories = [item["category"] for item in st.session_state.trash_items]
    logger.info(f"Generating map for trash categories: {trash_categories}")

    # Check if we have a cached route for current items and transport mode
    cache_key = f"{','.join(sorted(trash_categories))}_{st.session_state.transport_mode}"

    # Fun recycling tips and facts
    recycling_tips = [
        "♻️ Did you know? Recycling one aluminum can saves enough energy to power a TV for 3 hours!",
        "🌍 Every ton of recycled paper saves 17 trees, 7,000 gallons of water, and 463 gallons of oil.",
        "🔋 A single recycled battery can power a laptop for up to 15 hours!",
        "🍾 Glass can be recycled endlessly without losing quality or purity.",
        "📦 Cardboard boxes can be recycled 5-7 times before the fibers become too short.",
        "🌱 Composting food waste reduces methane emissions from landfills by up to 50%.",
        "💡 Recycling plastic bottles saves twice as much energy as burning them in an incinerator.",
        "🚴 Choosing to recycle is like biking instead of driving - every small action counts!",
        "🌊 Recycling helps reduce ocean pollution - 8 million tons of plastic enter oceans yearly.",
        "⚡ Recycling aluminum uses 95% less energy than creating new aluminum from raw materials!"
    ]

    # Calculate route with progress tracking
    import random
    import time

    tip = random.choice(recycling_tips)

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    # Show initial progress
    progress_placeholder.progress(0)
    status_placeholder.info(f"🔍 Finding optimal drop-off locations...\n\n{tip}")

    logger.info("Starting route calculation")

    # Progress callback function
    def update_progress(progress_value, status_message):
        progress_placeholder.progress(int(progress_value * 100))
        status_placeholder.info(f"{status_message}\n\n{tip}")

    # Step 1: Check cache or calculate
    route_segments, ordered_waypoints = None, None

    # Use cached route if available and matches current state
    if (st.session_state.cached_route and
        st.session_state.cached_route.get("cache_key") == cache_key):
        logger.info("Using cached route")
        route_segments = st.session_state.cached_route["route_segments"]
        ordered_waypoints = st.session_state.cached_route["ordered_waypoints"]
        progress_placeholder.empty()
        status_placeholder.empty()
    else:
        # Calculate new route
        try:
            update_progress(0.1, "🗺️ Starting route calculation...")

            route_segments, ordered_waypoints = calculate_route(
                st.session_state.user_location,
                trash_categories,
                st.session_state.transport_mode,
                progress_callback=update_progress
            )

            # Cache the result
            if route_segments and ordered_waypoints:
                st.session_state.cached_route = {
                    "cache_key": cache_key,
                    "route_segments": route_segments,
                    "ordered_waypoints": ordered_waypoints
                }
                logger.info("Route cached successfully")

            progress_placeholder.progress(100)
            status_placeholder.success("✅ Route calculated successfully!")
            time.sleep(0.8)

        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"Error calculating route: {e}")
            logger.exception(f"Route calculation failed: {e}")

        finally:
            progress_placeholder.empty()
            status_placeholder.empty()

    if route_segments and ordered_waypoints:
        logger.debug(f"Route calculated with {len(route_segments)} segments")

        # Calculate totals
        total_distance = sum(seg["distance"] for seg in route_segments)
        total_duration = sum(seg["duration"] for seg in route_segments)

        # Show route summary
        st.success(f"🎯 Route: {len(ordered_waypoints)} stops • {total_distance/1000:.1f} km • {total_duration//60} min")

        # Two-column layout: Map on left, drop-off list on right
        col_map, col_list = st.columns([3, 2])

        with col_map:
            st.markdown("### 🗺️ Route Map")

            # Calculate center and zoom to fit all points
            all_lats = [st.session_state.user_location["lat"]] + [w["lat"] for w in ordered_waypoints]
            all_lons = [st.session_state.user_location["lon"]] + [w["lon"] for w in ordered_waypoints]
            center_lat = sum(all_lats) / len(all_lats)
            center_lon = sum(all_lons) / len(all_lons)

            # Calculate zoom level based on bounding box
            lat_range = max(all_lats) - min(all_lats)
            lon_range = max(all_lons) - min(all_lons)
            max_range = max(lat_range, lon_range)

            # Dynamic zoom: tighter zoom for closer points
            if max_range < 0.01:  # Very close points (< 1km)
                zoom_level = 15
            elif max_range < 0.02:  # Close points (< 2km)
                zoom_level = 14
            elif max_range < 0.05:  # Medium distance (< 5km)
                zoom_level = 13
            elif max_range < 0.1:  # Further points (< 10km)
                zoom_level = 12
            else:  # Wide spread
                zoom_level = 11

            # Create Folium map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_level,
                tiles="OpenStreetMap"
            )

            # Add start point marker (green)
            folium.Marker(
                location=[st.session_state.user_location["lat"], st.session_state.user_location["lon"]],
                popup="<b>🏁 Start</b><br>Your Location",
                tooltip="Start",
                icon=folium.Icon(color='green', icon='home', prefix='fa')
            ).add_to(m)

            # Add route lines first (so they appear under markers)
            for segment in route_segments:
                if segment.get("geojson") and "features" in segment["geojson"]:
                    try:
                        # Extract coordinates from GeoJSON
                        coords = segment["geojson"]["features"][0]["geometry"]["coordinates"]
                        # Convert from [lon, lat] to [lat, lon] for Folium
                        route_coords = [[lat, lon] for lon, lat in coords]

                        # Add polyline for route
                        folium.PolyLine(
                            route_coords,
                            color='#3296FF',
                            weight=5,
                            opacity=0.8,
                            tooltip=f"{segment['distance']/1000:.1f} km • {segment['duration']//60} min"
                        ).add_to(m)
                    except (KeyError, IndexError) as e:
                        logger.warning(f"Could not render route geometry: {e}")

            # Add drop-off point markers (orange)
            for idx, waypoint in enumerate(ordered_waypoints, 1):
                icon_emoji = get_category_icon(waypoint['trash_class'])
                category_name = waypoint['trash_class'].replace('_', ' ').title()

                # Find matching trash items for this waypoint
                matching_items = [
                    item for item in st.session_state.trash_items
                    if item["category"] == waypoint['trash_class']
                ]

                # Create detailed popup with item thumbnails
                popup_html = f"""
                <div style="font-family: sans-serif; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0; color: #2E7D32;">{icon_emoji} Stop {idx}</h4>
                    <b>{category_name}</b><br>
                    <div style="margin: 8px 0;">
                        <b>📍 {waypoint.get('location_name', 'Collection Point')}</b><br>
                        <small style="color: #666;">{waypoint.get('address', 'Address unavailable')}</small>
                    </div>
                """

                # Add matching items info to popup
                if matching_items:
                    popup_html += f"""
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #ddd;">
                        <b>Items to drop:</b><br>
                    """
                    for item in matching_items:
                        conf_pct = item['confidence'] * 100
                        popup_html += f"""
                        <div style="margin: 5px 0;">
                            • {icon_emoji} {category_name}
                            <small style="color: #666;">({conf_pct:.0f}% confidence)</small>
                        </div>
                        """
                    popup_html += "</div>"

                popup_html += "</div>"

                # Enhanced tooltip with address
                location_name = waypoint.get('location_name', 'Collection Point')
                address_short = waypoint.get('address', '').split(',')[0] if waypoint.get('address') else ''
                tooltip_text = f"{icon_emoji} {category_name}"
                if address_short:
                    tooltip_text += f" - {address_short}"

                folium.Marker(
                    location=[waypoint["lat"], waypoint["lon"]],
                    popup=folium.Popup(popup_html, max_width=350),
                    tooltip=tooltip_text,
                    icon=folium.Icon(color='orange', icon='trash', prefix='fa')
                ).add_to(m)

            # Fit bounds to show all markers
            bounds = [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]]
            m.fit_bounds(bounds, padding=[30, 30])

            # Render Folium map (returned_objects=[] prevents rerun on map interaction)
            st_folium(m, width=None, height=500, use_container_width=True, returned_objects=[])

        with col_list:
            # Header with global Google Maps navigation
            col_header, col_nav = st.columns([2, 1])
            with col_header:
                st.markdown("### 📍 Locations")
            with col_nav:
                # Build waypoints string for multi-stop Google Maps route
                waypoints_str = "|".join([f"{w['lat']},{w['lon']}" for w in ordered_waypoints])
                gmaps_full_route = f"https://www.google.com/maps/dir/?api=1&origin={st.session_state.user_location['lat']},{st.session_state.user_location['lon']}&destination={ordered_waypoints[-1]['lat']},{ordered_waypoints[-1]['lon']}&waypoints={waypoints_str}"
                st.markdown(f"[🗺️ Navigate]({gmaps_full_route})")

            for seg_idx, segment in enumerate(route_segments, 1):
                # Find items for this destination
                destination_name = segment["to"]
                items_for_stop = []
                for item in st.session_state.trash_items:
                    for waypoint in ordered_waypoints:
                        if waypoint["name"] == destination_name and waypoint.get("trash_class") == item["category"]:
                            items_for_stop.append(item)

                # Get waypoint for address
                waypoint = next((w for w in ordered_waypoints if w["name"] == destination_name), None)

                # Distance and duration
                dist_km = segment["distance"] / 1000
                dur_min = segment["duration"] // 60

                # Build compact summary for expander header
                if items_for_stop:
                    items_text = ", ".join([f"{get_category_icon(item['category'])} {item['category'].replace('_', ' ').title()}" for item in items_for_stop])
                    summary = f"**{seg_idx}.** {items_text} • {dist_km:.1f}km • {dur_min}min"
                else:
                    summary = f"**{seg_idx}.** {dist_km:.1f}km • {dur_min}min"

                # Collapsible expander for details
                with st.expander(summary, expanded=False):
                    # Location name
                    if waypoint and waypoint.get("location_name"):
                        st.markdown(f"📍 **{waypoint.get('location_name')}**")

                    # Address
                    if waypoint and waypoint.get("address"):
                        st.caption(waypoint.get('address'))
    else:
        st.warning("Could not calculate route. Please check your connection and try again.")

    # Back button only
    st.markdown("---")
    if st.button("⬅️ Edit Items", key="map_back", use_container_width=False):
        logger.info("User clicked back from map screen to edit items")
        st.session_state.screen = 2
        st.rerun()


# ============================================================================
# MAIN APP NAVIGATION
# ============================================================================

def main():
    """Main app with screen routing."""
    logger.debug(f"Main function called - Current screen: {st.session_state.screen}")

    # CSS for better styling
    st.markdown("""
    <style>
        .stButton > button {
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        h1, h2 {
            color: #2E7D32;
        }
    </style>
    """, unsafe_allow_html=True)

    # Route to appropriate screen
    if st.session_state.screen == 0:
        screen_welcome()
    elif st.session_state.screen == 1:
        screen_location()
    elif st.session_state.screen == 2:
        screen_trash_items()
    elif st.session_state.screen == 3:
        screen_map()

    logger.debug(f"Screen render completed - Screen: {st.session_state.screen}")


if __name__ == "__main__":
    main()
