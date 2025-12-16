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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set GCP credentials path
GCP_CREDENTIALS_PATH = "/Users/dariaserbichenko/code/DariaSerb/key-gcp/trash-optimizer-479913-91e59ecc96c9.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH

# BigQuery configuration
PROJECT = "trash-optimizer-479913"
DATASET = "nantes"

# API endpoints
NANTES_BASE_URL_V1 = "https://data.nantesmetropole.fr/api/records/1.0/search/"
NANTES_BASE_URL_V2 = "https://data.nantesmetropole.fr/api/explore/v2.1/catalog/datasets"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_duplicates(df, strategy='coordinates'):
    """
    Remove duplicates based on different strategies

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str
        'coordinates', 'address', or 'strict'

    Returns:
    --------
    pd.DataFrame
        Deduplicated DataFrame
    """
    df_clean = df.copy()

    # Extract coordinates
    df_clean['lat'] = df_clean['geo_point_2d'].apply(lambda x: round(x[0], 6) if isinstance(x, list) else None)
    df_clean['lon'] = df_clean['geo_point_2d'].apply(lambda x: round(x[1], 6) if isinstance(x, list) else None)
    df_clean['adresse_clean'] = df_clean['adresse'].str.lower().str.strip()

    # Choose deduplication strategy
    if strategy == 'coordinates':
        # Keep first entry for each unique coordinate
        df_deduped = df_clean.drop_duplicates(subset=['lat', 'lon'], keep='first')

    elif strategy == 'address':
        # Keep first entry for each unique address/commune
        df_deduped = df_clean.drop_duplicates(subset=['adresse_clean', 'commune'], keep='first')

    elif strategy == 'strict':
        # Keep first entry for exact matches (excluding geo_point_2d list)
        cols = [col for col in df_clean.columns if col not in ['geo_point_2d']]
        df_deduped = df_clean.drop_duplicates(subset=cols, keep='first')

    else:
        raise ValueError("Strategy must be 'coordinates', 'address', or 'strict'")

    # Clean up temporary columns
    df_deduped = df_deduped.drop(columns=['lat', 'lon', 'adresse_clean'], errors='ignore')

    print(f"Original rows: {len(df)}")
    print(f"After {strategy} deduplication: {len(df_deduped)}")
    print(f"Removed {len(df) - len(df_deduped)} duplicates")

    return df_deduped


def clean_dataframe_for_bq(df_input):
    """
    Basic cleaning for BigQuery, preserving coordinate structure

    Parameters:
    -----------
    df_input : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame ready for BigQuery upload
    """
    df_clean_bq = df_input.copy()

    print("🧹 Cleaning DataFrame for BigQuery...")

    # 1. Fix column names
    original_cols = df_clean_bq.columns.tolist()
    df_clean_bq.columns = [str(col).replace(' ', '_').replace('-', '_').replace('.', '_').lower()
                          for col in df_clean_bq.columns]

    print(f"   Renamed columns: {dict(zip(original_cols, df_clean_bq.columns))}")

    # 2. Handle geo_point_2d - keep as string or extract coordinates
    if 'geo_point_2d' in df_clean_bq.columns:
        print("   Processing geo_point_2d column...")

        # Option 1: Keep as string (if you want to preserve the list structure as text)
        df_clean_bq['geo_point_2d_str'] = df_clean_bq['geo_point_2d'].astype(str)

        # Option 2: Extract latitude and longitude as separate columns
        try:
            df_clean_bq['latitude'] = df_clean_bq['geo_point_2d'].apply(
                lambda x: float(x[0]) if isinstance(x, list) and len(x) > 0 else None
            )
            df_clean_bq['longitude'] = df_clean_bq['geo_point_2d'].apply(
                lambda x: float(x[1]) if isinstance(x, list) and len(x) > 1 else None
            )
            print(f"   Extracted coordinates: {df_clean_bq['latitude'].notna().sum()} valid lat/lon pairs")
        except Exception as e:
            print(f"   Warning: Could not extract coordinates: {e}")

    # 3. Convert other lists/dicts to strings
    for col in df_clean_bq.columns:
        if col != 'geo_point_2d':  # Skip the original geo_point_2d
            if df_clean_bq[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                df_clean_bq[col] = df_clean_bq[col].astype(str)
                print(f"   Converted {col} to string (contains lists/dicts)")

    # 4. Fill NaN values
    for col in df_clean_bq.columns:
        if df_clean_bq[col].dtype == 'object':
            df_clean_bq[col] = df_clean_bq[col].fillna('')
        elif pd.api.types.is_numeric_dtype(df_clean_bq[col]):
            # For numeric columns, you might want to keep NaN or fill with 0
            # df_clean_bq[col] = df_clean_bq[col].fillna(0)  # Uncomment if needed
            pass

    # 5. Remove the original list column if we created string version
    if 'geo_point_2d' in df_clean_bq.columns and 'geo_point_2d_str' in df_clean_bq.columns:
        df_clean_bq = df_clean_bq.drop(columns=['geo_point_2d'])
        print("   Dropped original geo_point_2d column (kept string version)")

    print(f"   Final columns: {list(df_clean_bq.columns)}")

    return df_clean_bq


def upload_to_bigquery(df, table_name, project=PROJECT, dataset=DATASET):
    """
    Upload DataFrame to BigQuery

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to upload
    table_name : str
        Name of the BigQuery table
    project : str
        GCP project ID
    dataset : str
        BigQuery dataset name
    """
    table_id = f"{project}.{dataset}.{table_name}"

    print("\n" + "=" * 60)
    print(f"UPLOADING TO BIGQUERY: {table_name}")
    print("=" * 60)

    # Initialize client
    client = bigquery.Client(project=project)
    print(f"✅ BigQuery client initialized successfully")

    # Check dataset
    dataset_ref = f"{project}.{dataset}"
    try:
        dataset_obj = client.get_dataset(dataset_ref)
        print(f"✅ Dataset '{dataset}' exists")
        print(f"   Location: {dataset_obj.location}")
    except NotFound:
        print(f"📁 Creating dataset '{dataset}'...")
        dataset_obj = bigquery.Dataset(dataset_ref)
        dataset_obj.location = "EU"
        dataset_obj = client.create_dataset(dataset_obj, timeout=30)
        print(f"✅ Dataset created")
        print(f"   Location: {dataset_obj.location}")

    # Display DataFrame info
    print(f"\n📊 Data to upload:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Prepare DataFrame - ensure no lists/dicts
    df_clean = df.copy()

    # Clean column names for BigQuery compatibility
    df_clean.columns = df_clean.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    print(f"\n🧹 Cleaning data for BigQuery...")

    conversions = 0
    for col in df_clean.columns:
        # Convert lists/dicts to strings
        if df_clean[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
            df_clean[col] = df_clean[col].astype(str)
            conversions += 1
            print(f"   Converted {col} to string")

    # Fill NaN values for string columns
    nan_count = df_clean.isna().sum().sum()
    if nan_count > 0:
        print(f"   Found {nan_count} NaN values")
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('')

    print(f"   Cleaned shape: {df_clean.shape}")

    # Convert DataFrame to CSV in memory
    print("\n📄 Converting DataFrame to CSV in memory...")
    csv_buffer = io.StringIO()
    df_clean.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_content = csv_buffer.getvalue().encode('utf-8')

    # Create job configuration
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # Will replace existing table
        autodetect=True,                     # Let BigQuery detect schema
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,                 # Skip header row
        max_bad_records=100,                 # Allow some bad records
        encoding='UTF-8',
        allow_quoted_newlines=True
    )

    print(f"\n⬆️  Uploading {len(df_clean):,} rows to table '{table_name}'...")

    # Upload from CSV
    try:
        # Create file-like object
        file_obj = io.BytesIO(csv_content)

        # Submit job
        job = client.load_table_from_file(
            file_obj,
            table_id,
            job_config=job_config
        )

        print("   Job submitted. Waiting for completion...")
        job.result()  # Wait for completion

        # Verify upload
        table = client.get_table(table_id)
        print(f"\n✅ SUCCESS!")
        print(f"   Table: {table_id}")
        print(f"   Rows uploaded: {table.num_rows:,}")
        print(f"   Table size: {table.num_bytes / (1024*1024):.2f} MB")
        print(f"   Created: {table.created.strftime('%Y-%m-%d %H:%M:%S')}")

        # Show schema preview
        print(f"\n📐 Schema preview (first 5 columns):")
        for i, field in enumerate(table.schema[:5], 1):
            print(f"   {i}. {field.name:20} : {field.field_type}")

        if len(table.schema) > 5:
            print(f"   ... and {len(table.schema) - 5} more columns")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

        # Try alternative method
        print("\n🔄 Trying alternative upload method...")
        try:
            # Try direct DataFrame upload
            direct_job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                autodetect=True,
                max_bad_records=100
            )

            direct_job = client.load_table_from_dataframe(df_clean, table_id, job_config=direct_job_config)
            direct_job.result()

            table = client.get_table(table_id)
            print(f"✅ Direct upload successful!")
            print(f"   Rows uploaded: {table.num_rows:,}")

        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            print(f"\n💡 You can check the saved CSV file and upload it manually via Google Cloud Console")


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
for strategy in ['coordinates', 'address', 'strict']:
    df_test = clean_duplicates(df, strategy=strategy)
    print()

# Apply coordinate-based deduplication
print("Original shape:", df.shape)
df_clean = clean_duplicates(df, strategy='coordinates')
print("\nCleaned data shape:", df_clean.shape)
print("\nFirst few rows of cleaned data:")
print(df_clean[['adresse', 'commune', 'geo_point_2d']].head())

# Extract coordinates (geo_point_2d is a list of two numbers [lat, lon])
df_clean['lat'] = df_clean['geo_point_2d'].apply(lambda x: float(x[0]) if isinstance(x, list)
                                                  and len(x) > 0 else None)
df_clean['lon'] = df_clean['geo_point_2d'].apply(lambda x: float(x[1]) if isinstance(x, list)
                                                  and len(x) > 1 else None)
print("Successfully extracted coordinates as [lat, lon]")

# Save to CSV
df_clean.to_csv('alimentary_garbage.csv', index=False)
print("DataFrame saved as 'alimentary_garbage.csv'")

# Upload to BigQuery
df_bq_ready = clean_dataframe_for_bq(df_clean)
upload_to_bigquery(df_bq_ready, 'alimentary_garbage_clean')


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

# Extract coordinates from dictionaries
df1['lon'] = df1['geo_point_2d'].apply(
    lambda x: float(x['lon']) if isinstance(x, dict) and 'lon' in x else None
)
df1['lat'] = df1['geo_point_2d'].apply(
    lambda x: float(x['lat']) if isinstance(x, dict) and 'lat' in x else None
)
print(f"Successfully extracted coordinates for {df1['lon'].notna().sum()} rows")

# Save to CSV
df1.to_csv('ecopoints.csv', index=False)
print("DataFrame saved as 'ecopoints.csv'")

# Upload to BigQuery
upload_to_bigquery(df1, 'ecopoints')


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

# Extract coordinates
df2[['lat', 'lon']] = df2['position'].str.split(',', expand=True).astype(float)
print(f"Successfully extracted coordinates for {df2['lon'].notna().sum()} rows")

# Save to CSV
df2.to_csv('collection_centres_PdL_region.csv', index=False)
print("DataFrame saved as 'collection_centres_PdL_region.csv'")

# Upload to BigQuery
upload_to_bigquery(df2, 'collection_centres_pdl')


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

# Extract coordinates from dictionaries
df3['lon'] = df3['geo_point_2d'].apply(
    lambda x: float(x['lon']) if isinstance(x, dict) and 'lon' in x else None
)
df3['lat'] = df3['geo_point_2d'].apply(
    lambda x: float(x['lat']) if isinstance(x, dict) and 'lat' in x else None
)
print(f"Successfully extracted coordinates for {df3['lon'].notna().sum()} rows")

# Save to CSV
df3.to_csv('location_dropoff_points_nantes.csv', index=False)
print("DataFrame saved as 'location_dropoff_points_nantes.csv'")

# Upload to BigQuery
upload_to_bigquery(df3, 'location_dropoff_points_nantes')


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
