"""
Le Relais Collection Points Extractor
=====================================
Extracts textile collection points from Le Relais (https://lerelais.org)
by querying their AJAX endpoint with geographic bounds.
"""

import requests
import pandas as pd
import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import time
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "https://siroco.lerelais.org"
AJAX_ENDPOINT = f"{BASE_URL}/index.php"
DATA_DIR = os.getenv('DATA_DIR', '.')

# Geographic bounds for Pays de la Loire region
PAYS_DE_LA_LOIRE_BOUNDS = {
    'south_lat': 46.2,   # Southern limit (below Vendée)
    'south_lng': -2.6,   # Western limit (Atlantic coast)
    'north_lat': 48.6,   # Northern limit (above Mayenne)
    'north_lng': 0.9     # Eastern limit (toward Centre-Val de Loire)
}

# Geographic bounds for Nantes Metropole (more focused)
NANTES_METROPOLE_BOUNDS = {
    'south_lat': 47.1,   # South of Nantes
    'south_lng': -1.8,   # West of Nantes
    'north_lat': 47.4,   # North of Nantes
    'north_lng': -1.3    # East of Nantes
}

# Grid divisions - adjust based on desired coverage vs. speed
GRID_DIVISIONS_REGION = 8   # 8x8 = 64 queries for Pays de la Loire
GRID_DIVISIONS_METRO = 4    # 4x4 = 16 queries for Nantes Metropole only


def generate_grid_bounds(bounds: Dict, divisions: int) -> List[Tuple[float, float, float, float]]:
    """
    Generate a grid of geographic bounds to query the entire area.

    Args:
        bounds: Dict with south_lat, south_lng, north_lat, north_lng
        divisions: Number of divisions per axis

    Returns:
        List of tuples (south_lat, south_lng, north_lat, north_lng)
    """
    lat_step = (bounds['north_lat'] - bounds['south_lat']) / divisions
    lng_step = (bounds['north_lng'] - bounds['south_lng']) / divisions

    grid = []
    for i in range(divisions):
        for j in range(divisions):
            grid.append((
                bounds['south_lat'] + i * lat_step,
                bounds['south_lng'] + j * lng_step,
                bounds['south_lat'] + (i + 1) * lat_step,
                bounds['south_lng'] + (j + 1) * lng_step
            ))

    return grid


def fetch_collection_points(south_lat: float, south_lng: float,
                           north_lat: float, north_lng: float,
                           zoom: int = 10) -> Optional[Dict]:
    """
    Query Le Relais AJAX endpoint for collection points in given bounds.

    Args:
        south_lat: Southern latitude bound
        south_lng: Southern longitude bound
        north_lat: Northern latitude bound
        north_lng: Northern longitude bound
        zoom: Map zoom level (affects result density)

    Returns:
        JSON response dict or None on error
    """
    params = {
        'do': 'pointsdecollecte/ajax'
    }

    data = {
        'action': 'getPaLci',  # Required: specifies which collection points to fetch
        'nomPA': '',
        'adressePA': '',
        'zoom': zoom,
        'slat': south_lat,
        'slng': south_lng,
        'nlat': north_lat,
        'nlng': north_lng,
        'startPosLat': None,
        'startPosLng': None
    }

    try:
        response = requests.post(
            AJAX_ENDPOINT,
            params=params,
            data=data,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-Requested-With': 'XMLHttpRequest'
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for bounds ({south_lat}, {south_lng}) to ({north_lat}, {north_lng}): {e}")
        return None


def parse_collection_points(response_data: Dict) -> List[Dict]:
    """
    Parse collection point data from API response.

    Args:
        response_data: JSON response from Le Relais API

    Returns:
        List of collection point dictionaries
    """
    if not response_data or response_data.get('status') != 1:
        return []

    points = []
    # Updated: API returns data in 'aappEtEmpPourCarte' array
    locations_data = response_data.get('aappEtEmpPourCarte', [])

    for location in locations_data:
        # Extract address from HTML content
        content = location.get('content', '')
        address = ''
        city = ''
        postal_code = ''

        if content:
            # Extract address using simple string parsing
            # Address is in format: <div class="lci_address">\n address<br>\n postal_code city<br><br>
            import re

            # Find address block
            address_match = re.search(r'<div class="lci_address">(.*?)</div>', content, re.DOTALL)
            if address_match:
                address_block = address_match.group(1)

                # Remove HTML tags and clean up
                address_lines = re.sub(r'<[^>]+>', '', address_block).strip().split('\n')
                address_lines = [line.strip() for line in address_lines if line.strip()]

                if len(address_lines) >= 2:
                    # First line is usually the street address
                    address = address_lines[0]

                    # Second line usually contains postal code and city
                    if len(address_lines) >= 2:
                        postal_city = address_lines[1]
                        # Extract postal code (5 digits) and city
                        postal_match = re.match(r'(\d{5})\s+(.+)', postal_city)
                        if postal_match:
                            postal_code = postal_match.group(1)
                            city = postal_match.group(2)

        point = {
            'name': location.get('name', ''),
            'address': address if address else 'Address not specified',
            'city': city,
            'postal_code': postal_code,
            'latitude': location.get('lat'),
            'longitude': location.get('lng'),
            'id': location.get('id', ''),
            'idPA': location.get('idPA', ''),
            'inactive': location.get('inactif', False)
        }
        points.append(point)

    return points


def extract_all_lerelais_points(bounds: Dict = NANTES_METROPOLE_BOUNDS,
                                divisions: int = GRID_DIVISIONS_METRO,
                                delay: float = 1.0) -> pd.DataFrame:
    """
    Extract all Le Relais collection points by querying in a grid pattern.

    Args:
        bounds: Geographic bounds to query
        divisions: Number of grid divisions per axis
        delay: Delay between requests in seconds

    Returns:
        DataFrame with all collection points
    """
    print("=" * 60)
    print("Le Relais Collection Points Extraction")
    print("=" * 60)

    grid = generate_grid_bounds(bounds, divisions)
    print(f"\nQuerying {len(grid)} grid cells...")

    all_points = []
    seen_ids = set()

    for idx, (slat, slng, nlat, nlng) in enumerate(grid, 1):
        print(f"\nQuery {idx}/{len(grid)}: ({slat:.2f}, {slng:.2f}) to ({nlat:.2f}, {nlng:.2f})")

        response = fetch_collection_points(slat, slng, nlat, nlng)

        if response:
            points = parse_collection_points(response)

            # Deduplicate by ID
            new_points = [p for p in points if p['id'] not in seen_ids]
            seen_ids.update(p['id'] for p in new_points)
            all_points.extend(new_points)

            print(f"  Found {len(points)} points ({len(new_points)} new, {len(all_points)} total)")

        # Rate limiting
        if idx < len(grid):
            time.sleep(delay)

    print(f"\n{'=' * 60}")
    print(f"Total unique collection points found: {len(all_points)}")
    print(f"{'=' * 60}\n")

    return pd.DataFrame(all_points)


def save_to_csv(df: pd.DataFrame, filename: str = 'lerelais_collection_points.csv') -> str:
    """
    Save collection points to CSV file.

    Args:
        df: DataFrame with collection points
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_path = Path(DATA_DIR) / filename
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved {len(df)} collection points to: {output_path}")
    return str(output_path)


def main():
    """Main execution function."""

    # Choose your target area
    # Option 1: Nantes Metropole only (faster, focused) - DEFAULT
    bounds = NANTES_METROPOLE_BOUNDS
    divisions = GRID_DIVISIONS_METRO
    area_name = "Nantes Metropole"

    # Option 2: Full Pays de la Loire region (uncomment to use)
    # bounds = PAYS_DE_LA_LOIRE_BOUNDS
    # divisions = GRID_DIVISIONS_REGION
    # area_name = "Pays de la Loire"

    print(f"\nStarting Le Relais collection points extraction for {area_name}...")
    print(f"Target area: lat: {bounds['south_lat']}-{bounds['north_lat']}, "
          f"lng: {bounds['south_lng']}-{bounds['north_lng']}")
    print(f"Grid: {divisions}x{divisions} = {divisions*divisions} queries\n")

    # Extract points
    df = extract_all_lerelais_points(bounds=bounds, divisions=divisions)

    if df.empty:
        print("\nNo collection points found!")
        return

    # Display summary statistics
    print("\nDataset Summary:")
    print("-" * 60)
    print(f"Total points: {len(df)}")
    print(f"Points with coordinates: {df['latitude'].notna().sum()}")
    if 'city' in df.columns:
        cities_count = df[df['city'].notna() & (df['city'] != '')]['city'].nunique()
        print(f"Cities covered: {cities_count}")
    print(f"Inactive points: {df['inactive'].sum() if 'inactive' in df.columns else 0}")

    # Save to CSV
    output_file = save_to_csv(df)

    # Display sample data
    print("\nSample data:")
    print(df.head(10).to_string())

    print(f"\n✅ Extraction complete! Data saved to: {output_file}")


if __name__ == "__main__":
    main()
