#!/usr/bin/env python3
"""
Extract Garbage Collection Location Data for Nantes

This script extracts and processes waste collection location data from multiple sources:
1. Food waste collection points (biodéchets)
2. Recycling centers and eco-points
3. Regional collection centers (Pays de la Loire)
4. Voluntary drop-off columns

The script performs the following operations:
- Downloads data from Nantes Métropole open data API
- Cleans and deduplicates records
- Extracts geographic coordinates
- Uploads processed data to Google BigQuery
- Creates a unified table combining all sources

Data Sources:
- Nantes Métropole Open Data API
- Pays de la Loire Regional Data

Output:
- Individual CSV files for each data source
- BigQuery tables: alimentary_garbage_clean, ecopoints, collection_centres_pdl, location_dropoff_points_nantes
- Unified table: all_trash_locations
"""

import requests
import os
import io
import sys
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest
from dotenv import load_dotenv
import bigquery_utils as bq_utils

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
GCP_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
PROJECT = os.getenv('GCP_PROJECT')
DATASET = os.getenv('GCP_DATASET')
DATA_DIR = os.getenv('DATA_DIR', '.')

# Validate required environment variables
if not all([GCP_CREDENTIALS_PATH, PROJECT, DATASET]):
    raise ValueError(
        "Missing required environment variables. Please check your .env file.\n"
        "Required: GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT, GCP_DATASET\n"
        "Copy .env.template to .env and fill in your values."
    )

# Set GCP credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH

# API endpoints
NANTES_BASE_URL_V1 = "https://data.nantesmetropole.fr/api/records/1.0/search/"
NANTES_BASE_URL_V2 = "https://data.nantesmetropole.fr/api/explore/v2.1/catalog/datasets"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Note: clean_duplicates function now provided by bigquery_utils module


# Note: clean_dataframe_for_bq function now provided by bigquery_utils module


# Note: upload_to_bigquery function now provided by bigquery_utils module


# =============================================================================
# SECTION 1: Food Waste Collection Points (Biodéchets)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 1: Food Waste Collection Points")
print("=" * 60)

params = {
    "dataset": "244400404_point-collecte-dechets-alimentaires-biodechet-nantes",
    "rows": 10000
}

response = requests.get(NANTES_BASE_URL_V1, params=params)
response.raise_for_status()
data = response.json()

# Extract records
records = []
for record in data['records']:
    fields = record['fields']
    records.append(fields)

df = pd.DataFrame(records)
print(f"Total records: {len(df)}")
print(df.head())

# Check data structure
print(f"\nType of first element: {type(df['geo_point_2d'][0])}")
print(f"First element: {df['geo_point_2d'][0]}")
print(f"First element length: {len(df['geo_point_2d'][0])}")

# Try different deduplication strategies
print("\n=== DEDUPLICATION STRATEGIES ===\n")

# First extract coordinates for deduplication
df_with_coords = bq_utils.extract_coordinates_from_dict(df, 'geo_point_2d')

for strategy in ['coordinates', 'address', 'strict']:
    df_test = bq_utils.clean_duplicates(df_with_coords, strategy=strategy)
    print()

# Apply coordinate-based deduplication
print("Original shape:", df.shape)
df_clean = bq_utils.clean_duplicates(df_with_coords, strategy='coordinates')
print("\nCleaned data shape:", df_clean.shape)
print("\nFirst few rows of cleaned data:")
print(df_clean[['adresse', 'commune', 'geo_point_2d']].head())

# Coordinates already extracted during deduplication
print(f"Coordinates available: {df_clean['lat'].notna().sum()} rows with valid lat/lon")

# Save to CSV
csv_file = os.path.join(DATA_DIR, 'alimentary_garbage.csv')
df_clean.to_csv(csv_file, index=False)
print(f"DataFrame saved as '{csv_file}'")

# Upload to BigQuery using utility function
bq_utils.upload_dataframe_to_bigquery(
    df_clean,
    'alimentary_garbage_clean',
    PROJECT,
    DATASET
)


# =============================================================================
# SECTION 2: Recycling Centers and Eco-points
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Recycling Centers and Eco-points")
print("=" * 60)

BASE_URL = f"{NANTES_BASE_URL_V2}/244400404_decheteries-ecopoints-nantes-metropole/records"

# To get all records, we use limit=-1
params = {"limit": -1}

response = requests.get(BASE_URL, params=params)
response.raise_for_status()
data = response.json()

records = data.get('results', [])
print(f"Total records retrieved: {len(records)}")

df1 = pd.DataFrame(response.json()['results'])
print(df1.head(5))
print("\nColumns:", df1.columns)

# Check ID uniqueness
print("\nChecking ID uniqueness (identifiant column):")
if 'identifiant' in df1.columns:
    total_ids = len(df1['identifiant'])
    unique_ids = df1['identifiant'].nunique()
    duplicate_id_count = df1['identifiant'].duplicated().sum()

    print(f"Total IDs: {total_ids}")
    print(f"Unique IDs: {unique_ids}")
    print(f"Duplicate IDs: {duplicate_id_count}")

    if duplicate_id_count > 0:
        print("DUPLICATE IDs FOUND:")
        dup_ids = df1[df1['identifiant'].duplicated(keep=False)]
        for id_val in dup_ids['identifiant'].unique():
            id_rows = dup_ids[dup_ids['identifiant'] == id_val]
            print(f"\nID {id_val} appears {len(id_rows)} times:")
            for _, row in id_rows.iterrows():
                print(f"  - {row.get('nom', 'Unknown')} | {row.get('adresse', 'No address')}")
else:
    print("No 'identifiant' column found")

# Extract coordinates using utility function
df1 = bq_utils.extract_coordinates_from_dict(df1, 'geo_point_2d')

# Save to CSV
csv_file = os.path.join(DATA_DIR, 'ecopoints.csv')
df1.to_csv(csv_file, index=False)
print(f"DataFrame saved as '{csv_file}'")

# Upload to BigQuery using utility function
bq_utils.upload_dataframe_to_bigquery(df1, 'ecopoints', PROJECT, DATASET)


# =============================================================================
# SECTION 3: Regional Collection Centers (Pays de la Loire)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: Regional Collection Centers")
print("=" * 60)

# Direct CSV download URL from the web page
CSV_URL = "https://data.nantesmetropole.fr/explore/dataset/837810944_annuairedesdecheteriesdma_pdl@data-teo-paysdelaloire/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B"

# Download the CSV
print("Downloading CSV data...")
response = requests.get(CSV_URL)
response.raise_for_status()

# Read CSV directly from the response content
df2 = pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=';')
print(f"Dataset loaded: {len(df2)} records")
print("\nColumns:", list(df2.columns))
print("\nFirst few rows:")
print(df2.head())

# Quick duplicate check
print("\nQUICK df2 DUPLICATE CHECK")
print(f"Total rows: {len(df2)}")
print(f"Exact duplicates: {df2.duplicated().sum()}")
print(f"Unique rows: {df2.drop_duplicates().shape[0]}")

# Check key columns
print("Key column duplicates:")
key_columns = ['C_SERVICE', 'N_SERVICE', 'AD1_SITE', 'GPS_LAT', 'GPS_LONG']
for col in key_columns:
    if col in df2.columns:
        dup_count = df2[col].duplicated().sum()
        unique_count = df2[col].nunique()
        print(f"  {col}: {dup_count} duplicates ({unique_count} unique values)")

# Extract coordinates (position column has lat,lon format)
if 'position' in df2.columns:
    df2[['lat', 'lon']] = df2['position'].str.split(',', expand=True).astype(float)
    print(f"Successfully extracted coordinates for {df2['lon'].notna().sum()} rows")

# Save to CSV
csv_file = os.path.join(DATA_DIR, 'collection_centres_PdL_region.csv')
df2.to_csv(csv_file, index=False)
print(f"DataFrame saved as '{csv_file}'")

# Upload to BigQuery using utility function
bq_utils.upload_dataframe_to_bigquery(df2, 'collection_centres_pdl', PROJECT, DATASET)


# =============================================================================
# SECTION 4: Voluntary Drop-off Columns (Colonnes d'apports volontaires)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: Voluntary Drop-off Columns")
print("=" * 60)

BASE_URL = f"{NANTES_BASE_URL_V2}/244400404_localisation-des-colonnes-apports-volontaires-de-nantes-metropole/records"

all_records = []
limit = 100  # Records per page
offset = 0
total_count = None

print("Fetching data with pagination...")
while True:
    # Build URL with current offset
    url = f"{BASE_URL}?limit={limit}&offset={offset}"
    print(f"  Fetching {limit} records from offset {offset}...")

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Get total count on first request
    if total_count is None:
        total_count = data.get('total_count', 0)
        print(f"  Total records available: {total_count}")

    # Add records from this page
    page_records = data.get('results', [])
    all_records.extend(page_records)

    # Update offset
    offset += limit

    # Stop if we have all records or if page is empty
    if not page_records or offset >= total_count:
        break

# Create DataFrame
df3 = pd.DataFrame(all_records)
print(f"\nTotal records fetched: {len(df3)}")
print("\nColumns:", df3.columns)

# Quick duplicate check
print("\nQUICK df3 DUPLICATE CHECK")

# Create a string version for duplicate checking
df3_str = df3.copy()

# Convert any dictionary columns to strings
for col in df3_str.columns:
    if df3_str[col].apply(lambda x: isinstance(x, dict)).any():
        df3_str[col] = df3_str[col].astype(str)

# Now check duplicates
print(f"Total rows: {len(df3)}")
print(f"Exact duplicates: {df3_str.duplicated().sum()}")
print(f"Unique rows: {df3_str.drop_duplicates().shape[0]}")

# Check key columns that ACTUALLY exist in df3
print("\nKey column duplicates (actual df3 columns):")
actual_key_columns = ['id_colonne', 'adresse', 'commune', 'type_dechet', 'type_colonne']
for col in actual_key_columns:
    if col in df3.columns:
        dup_count = df3[col].duplicated().sum()
        unique_count = df3[col].nunique()
        print(f"  {col}: {dup_count} duplicates ({unique_count} unique values)")

# Check geo_point_2d structure
print(f"\ngeo_point_2d sample: {df3['geo_point_2d'][0]}")

# Extract coordinates using utility function
df3 = bq_utils.extract_coordinates_from_dict(df3, 'geo_point_2d')

# Save to CSV
csv_file = os.path.join(DATA_DIR, 'location_dropoff_points_nantes.csv')
df3.to_csv(csv_file, index=False)
print(f"DataFrame saved as '{csv_file}'")

# Upload to BigQuery using utility function
bq_utils.upload_dataframe_to_bigquery(df3, 'location_dropoff_points_nantes', PROJECT, DATASET)


# =============================================================================
# SECTION 5: Create Unified Table
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: Creating Unified Table")
print("=" * 60)

UNIFIED_TABLE = "all_trash_locations"
unified_table_id = f"{PROJECT}.{DATASET}.{UNIFIED_TABLE}"

client = bigquery.Client(project=PROJECT)

# SQL query to create unified table
create_unified_table_query = f"""
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.{UNIFIED_TABLE}` AS

-- From alimentary_garbage_clean
SELECT
  'alimentary' as trash_type,
  identifiant as nom,
  adresse,
  lat as latitude,
  lon as longitude
FROM `{PROJECT}.{DATASET}.alimentary_garbage_clean`
WHERE lat IS NOT NULL AND lon IS NOT NULL

UNION ALL

-- From ecopoints
SELECT
  'ecopoints' as trash_type,
  nom,
  adresse,
  lat as latitude,
  lon as longitude
FROM `{PROJECT}.{DATASET}.ecopoints`
WHERE lat IS NOT NULL AND lon IS NOT NULL

UNION ALL

-- From collection_centres_pdl
SELECT
  'collection_centres' as trash_type,
  N_SERVICE as nom,
  AD1_ACTEUR as adresse,
  lat as latitude,
  lon as longitude
FROM `{PROJECT}.{DATASET}.collection_centres_pdl`
WHERE lat IS NOT NULL AND lon IS NOT NULL
"""

print("Creating unified table...")
try:
    # Execute query
    job = client.query(create_unified_table_query)
    job.result()
    print("✅ Unified table created successfully!")

    # Get table info
    table = client.get_table(unified_table_id)
    print(f"\nTable info:")
    print(f"   Name: {unified_table_id}")
    print(f"   Rows: {table.num_rows}")
    print(f"   Size: {table.num_bytes / (1024*1024):.2f} MB")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("SCRIPT COMPLETED")
print("=" * 60)
print("\nGenerated files:")
print("  - alimentary_garbage.csv")
print("  - ecopoints.csv")
print("  - collection_centres_PdL_region.csv")
print("  - location_dropoff_points_nantes.csv")
print("\nBigQuery tables created:")
print(f"  - {PROJECT}.{DATASET}.alimentary_garbage_clean")
print(f"  - {PROJECT}.{DATASET}.ecopoints")
print(f"  - {PROJECT}.{DATASET}.collection_centres_pdl")
print(f"  - {PROJECT}.{DATASET}.location_dropoff_points_nantes")
print(f"  - {PROJECT}.{DATASET}.all_trash_locations (unified)")
