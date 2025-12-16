"""
Unified Waste Collection Dataset Creation
=========================================
Creates and maintains a unified BigQuery dataset for trash collection points
with dynamic schema expansion and prioritized data ingestion.
"""

# install required libraries
# pip install google-cloud-bigquery pandas google-auth

import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

# ========================================================
# CONFIGURATION
# ========================================================

# Set up Google Cloud credentials
CREDENTIALS_PATH = "/Users/dariaserbichenko/code/DariaSerb/key-gcp/trash-optimizer-479913-91e59ecc96c9.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# BigQuery configuration
PROJECT = "trash-optimizer-479913"
DATASET = "nantes"
TARGET_TABLE = "trash_collection_points"

# ========================================================
# BIGQUERY CLIENT INITIALIZATION
# ========================================================

print("=" * 60)
print("INITIALIZING TRASH COLLECTION DATASET CREATION")
print("=" * 60)

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT)

# ========================================================
# 1. FOOD WASTE DATA (PRIORITY 1)
# ========================================================

print("\n1. PROCESSING FOOD WASTE DATA")
print("-" * 40)

query_food = f"""
SELECT
  ROW_NUMBER() OVER () as point_id,
  CONCAT('Food Waste - ', COALESCE(commune, 'Nantes')) as name,
  COALESCE(adresse, 'Address not specified') as address,
  lon as longitude,
  lat as latitude,
  1 as accepts_food,
  0 as accepts_cardboard,
  0 as accepts_glass,
  0 as accepts_metal,
  0 as accepts_paper,
  0 as accepts_plastic,
  0 as accepts_textile,
  0 as accepts_wood,
  0 as accepts_vegetation,
  'food_waste' as point_type,
  'Nantes_Metropole' as operator,
  CURRENT_DATE() as ingestion_date,
  '1.0' as schema_version,
  'alimentary_garbage_clean' as data_source
FROM `{PROJECT}.{DATASET}.alimentary_garbage_clean`
WHERE lat IS NOT NULL AND lon IS NOT NULL
"""

try:
    df_food = client.query(query_food).to_dataframe()
    print(f"✅ Retrieved {len(df_food):,} food waste locations")
except Exception as e:
    print(f"❌ Error retrieving food waste data: {e}")
    df_food = pd.DataFrame()

# ========================================================
# 2. RECYCLING CENTERS (PRIORITY 2)
# ========================================================

print("\n2. PROCESSING RECYCLING CENTERS")
print("-" * 40)

query_recycling = f"""
SELECT
  ROW_NUMBER() OVER () + 10000 as point_id,
  CONCAT('Recycling Center - ', COALESCE(nom, commune, 'Ecopoint')) as name,
  COALESCE(adresse, 'Address not specified') as address,
  lon as longitude,
  lat as latitude,
  0 as accepts_food,
  CASE WHEN UPPER(carton) = 'OUI' THEN 1 ELSE 0 END as accepts_cardboard,
  CASE WHEN UPPER(verre) = 'OUI' THEN 1 ELSE 0 END as accepts_glass,
  CASE WHEN UPPER(ferraille) = 'OUI' THEN 1 ELSE 0 END as accepts_metal,
  CASE WHEN UPPER(papier) = 'OUI' THEN 1 ELSE 0 END as accepts_paper,
  0 as accepts_plastic,
  CASE WHEN UPPER(textile) = 'OUI' THEN 1 ELSE 0 END as accepts_textile,
  CASE WHEN UPPER(bois) = 'OUI' THEN 1 ELSE 0 END as accepts_wood,
  CASE WHEN UPPER(dechet_vert) = 'OUI' THEN 1 ELSE 0 END as accepts_vegetation,
  'recycling_center' as point_type,
  'Nantes_Metropole' as operator,
  CURRENT_DATE() as ingestion_date,
  '1.0' as schema_version,
  'ecopoints' as data_source
FROM `{PROJECT}.{DATASET}.ecopoints`
WHERE lat IS NOT NULL AND lon IS NOT NULL
"""

try:
    df_recycling = client.query(query_recycling).to_dataframe()
    print(f"✅ Retrieved {len(df_recycling):,} recycling centers")

    # Show waste acceptance statistics
    waste_cols = [col for col in df_recycling.columns if col.startswith('accepts_')]
    print("   Waste acceptance in recycling centers:")
    for col in waste_cols:
        count = df_recycling[col].sum()
        if count > 0:
            waste_name = col.replace('accepts_', '').replace('_', ' ').title()
            print(f"   • {waste_name}: {count}/{len(df_recycling)} locations")

except Exception as e:
    print(f"❌ Error retrieving recycling centers: {e}")
    df_recycling = pd.DataFrame()

# ========================================================
# 3. GLASS COLLECTION POINTS (PRIORITY 3)
# ========================================================

print("\n3. PROCESSING GLASS COLLECTION POINTS")
print("-" * 40)

query_glass = f"""
SELECT
  ROW_NUMBER() OVER () + 30000 as point_id,
  CONCAT(
    'Glass Collection - ',
    COALESCE(
      CASE
        WHEN type_colonne IS NOT NULL THEN
          CASE type_colonne
            WHEN 'colonne enterrée' THEN 'Underground'
            WHEN 'colonne aérienne' THEN 'Above-ground'
            ELSE INITCAP(type_colonne)
          END
        ELSE ''
      END,
      'Glass Column'
    ),
    CASE
      WHEN commune IS NOT NULL THEN CONCAT(' - ', commune)
      ELSE ' - Nantes'
    END
  ) as name,
  COALESCE(adresse, 'Nantes Métropole') as address,
  lat as latitude,
  lon as longitude,
  0 as accepts_food,
  0 as accepts_cardboard,
  1 as accepts_glass,
  0 as accepts_metal,
  0 as accepts_paper,
  0 as accepts_plastic,
  0 as accepts_textile,
  0 as accepts_wood,
  0 as accepts_vegetation,
  'glass_column' as point_type,
  'Nantes_Metropole' as operator,
  CURRENT_DATE() as ingestion_date,
  '1.0' as schema_version,
  'location_dropoff_points_nantes' as data_source
FROM `{PROJECT}.{DATASET}.location_dropoff_points_nantes`
WHERE
  lat IS NOT NULL
  AND lon IS NOT NULL
  AND LOWER(TRIM(type_dechet)) = 'verre'
"""

try:
    df_glass = client.query(query_glass).to_dataframe()
    print(f"✅ Retrieved {len(df_glass):,} glass collection points")
except Exception as e:
    print(f"❌ Error retrieving glass collection points: {e}")
    df_glass = pd.DataFrame()

# ========================================================
# 4. OTHER WASTE TYPES (PRIORITY 4)
# ========================================================

print("\n4. PROCESSING OTHER WASTE TYPES")
print("-" * 40)

query_other_waste = f"""
SELECT
  ROW_NUMBER() OVER () +
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 40000
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 50000
    WHEN LOWER(TRIM(type_dechet)) LIKE '%ordure ménagère%' THEN 60000
    ELSE 70000
  END as point_id,

  CONCAT(
    CASE
      WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 'Paper/Cardboard'
      WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 'Recyclable Waste'
      WHEN LOWER(TRIM(type_dechet)) LIKE '%ordure ménagère%' THEN 'Household Waste'
      ELSE INITCAP(type_dechet)
    END,
    ' Collection - ',
    COALESCE(commune, 'Nantes'),
    CASE
      WHEN type_colonne IS NOT NULL THEN CONCAT(' (',
        CASE type_colonne
          WHEN 'colonne enterrée' THEN 'Underground'
          WHEN 'colonne aérienne' THEN 'Above-ground'
          ELSE INITCAP(type_colonne)
        END, ')')
      ELSE ''
    END
  ) as name,

  COALESCE(adresse, 'Nantes Métropole') as address,
  lat as latitude,
  lon as longitude,

  -- Waste acceptance capabilities
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 1
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as accepts_cardboard,

  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 1
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as accepts_paper,

  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as accepts_plastic,

  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as accepts_metal,

  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%ordure ménagère%' THEN 1
    ELSE 0
  END as accepts_miscellaneous,

  0 as accepts_food,
  0 as accepts_glass,
  0 as accepts_textile,
  0 as accepts_wood,
  0 as accepts_vegetation,

  'waste_column' as point_type,
  'Nantes_Metropole' as operator,
  CURRENT_DATE() as ingestion_date,
  '1.0' as schema_version,
  'location_dropoff_points_nantes' as data_source,

  type_dechet as original_waste_type,
  commune as city

FROM `{PROJECT}.{DATASET}.location_dropoff_points_nantes`
WHERE
  lat IS NOT NULL
  AND lon IS NOT NULL
  AND (
    LOWER(TRIM(type_dechet)) LIKE '%papier%carton%'
    OR LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%'
    OR LOWER(TRIM(type_dechet)) LIKE '%ordure ménagère%'
  )
"""

try:
    df_other_waste = client.query(query_other_waste).to_dataframe()
    print(f"✅ Retrieved {len(df_other_waste):,} other waste collection points")

    # Show breakdown
    if 'original_waste_type' in df_other_waste.columns:
        waste_counts = df_other_waste['original_waste_type'].value_counts()
        print("   Breakdown by waste type:")
        for waste_type, count in waste_counts.items():
            percentage = (count / len(df_other_waste)) * 100
            print(f"   • {waste_type}: {count:,} points ({percentage:.1f}%)")

except Exception as e:
    print(f"❌ Error retrieving other waste types: {e}")
    df_other_waste = pd.DataFrame()

# ========================================================
# 5. SPECIAL WASTE COLLECTION (ECOSYSTEM) - PRIORITY 5
# ========================================================

print("\n5. PROCESSING SPECIAL WASTE COLLECTION POINTS")
print("-" * 40)

# First, let's check what columns actually exist in the table
print("🔍 Checking table schema...")
try:
    # Get table schema
    table_ref = client.dataset(DATASET).table("ecosystem_collection_points_with_coords")
    table = client.get_table(table_ref)

    # List all columns
    existing_columns = [col.name for col in table.schema]
    print(f"✅ Table has {len(existing_columns)} columns")

    # Check for specific columns
    print("\n📋 Checking for waste type columns:")
    columns_to_check = [
        'Is_Neon_enabled', 'Is_Cartridge_enabled', 'Is_Lamp_Light_enabled',
        'Is_Battery_enabled', 'Is_Car_Battery_enabled', 'Is_Pile_enabled'
    ]

    for col in columns_to_check:
        if col in existing_columns:
            print(f"   ✓ {col} exists")
        else:
            print(f"   ✗ {col} NOT FOUND")

    # Build the query dynamically based on what columns exist
    select_columns = []
    select_columns.append("ID as point_id")
    select_columns.append("Name as name")
    select_columns.append("Address as address")
    select_columns.append("Longitude as longitude")
    select_columns.append("Latitude as latitude")

    # Add standard waste columns (always 0 for Ecosystem)
    select_columns.append("0 as accepts_food")
    select_columns.append("0 as accepts_cardboard")
    select_columns.append("0 as accepts_glass")
    select_columns.append("0 as accepts_metal")
    select_columns.append("0 as accepts_paper")
    select_columns.append("0 as accepts_plastic")
    select_columns.append("0 as accepts_textile")
    select_columns.append("0 as accepts_wood")
    select_columns.append("0 as accepts_vegetation")

    # Add special waste columns with COALESCE to handle missing columns
    if 'Is_Neon_enabled' in existing_columns:
        select_columns.append("Is_Neon_enabled as accepts_neon")
    else:
        select_columns.append("0 as accepts_neon")  # Default if column doesn't exist

    if 'Is_Cartridge_enabled' in existing_columns:
        select_columns.append("Is_Cartridge_enabled as accepts_cartridge")
    else:
        select_columns.append("0 as accepts_cartridge")

    if 'Is_Lamp_Light_enabled' in existing_columns:
        select_columns.append("Is_Lamp_Light_enabled as accepts_lamp")
    else:
        select_columns.append("0 as accepts_lamp")

    if 'Is_Battery_enabled' in existing_columns:
        select_columns.append("Is_Battery_enabled as accepts_battery")
    else:
        select_columns.append("0 as accepts_battery")

    if 'Is_Car_Battery_enabled' in existing_columns:
        select_columns.append("Is_Car_Battery_enabled as accepts_car_battery")
    else:
        select_columns.append("0 as accepts_car_battery")

    if 'Is_Pile_enabled' in existing_columns:
        select_columns.append("Is_Pile_enabled as accepts_pile")
    else:
        select_columns.append("0 as accepts_pile")

    # Add metadata columns
    select_columns.append("'special_waste' as point_type")
    select_columns.append("'Ecosystem' as operator")
    select_columns.append("CURRENT_DATE() as ingestion_date")
    select_columns.append("'1.0' as schema_version")
    select_columns.append("'ecosystem_collection_points_with_coords' as data_source")

    # Build the query
    query_special_waste = f"""
    SELECT
      {', '.join(select_columns)}
    FROM `{PROJECT}.{DATASET}.ecosystem_collection_points_with_coords`
    WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL
    """

    print(f"\n📝 Generated SQL query with {len(select_columns)} columns")

except Exception as e:
    print(f"❌ Error checking table schema: {e}")

    # Fallback query if we can't check schema
    query_special_waste = f"""
    SELECT
      ID as point_id,
      Name as name,
      Address as address,
      Longitude as longitude,
      Latitude as latitude,
      0 as accepts_food,
      0 as accepts_cardboard,
      0 as accepts_glass,
      0 as accepts_metal,
      0 as accepts_paper,
      0 as accepts_plastic,
      0 as accepts_textile,
      0 as accepts_wood,
      0 as accepts_vegetation,
      -- Try to get special waste columns, use 0 if they don't exist
      COALESCE(Is_Neon_enabled, 0) as accepts_neon,
      COALESCE(Is_Cartridge_enabled, 0) as accepts_cartridge,
      COALESCE(Is_Lamp_Light_enabled, 0) as accepts_lamp,
      COALESCE(Is_Battery_enabled, 0) as accepts_battery,
      COALESCE(Is_Car_Battery_enabled, 0) as accepts_car_battery,
      COALESCE(Is_Pile_enabled, 0) as accepts_pile,
      'special_waste' as point_type,
      'Ecosystem' as operator,
      CURRENT_DATE() as ingestion_date,
      '1.0' as schema_version,
      'ecosystem_collection_points_with_coords' as data_source
    FROM `{PROJECT}.{DATASET}.ecosystem_collection_points_with_coords`
    WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL
    """
    print("⚠️ Using fallback query with COALESCE")

try:
    df_special_waste = client.query(query_special_waste).to_dataframe()
    print(f"✅ Retrieved {len(df_special_waste):,} special waste collection points")

    # Show special waste types
    special_cols = [col for col in df_special_waste.columns if col.startswith('accepts_')]
    print("\n📊 Special waste acceptance:")
    for col in special_cols:
        count = df_special_waste[col].sum()
        if count > 0:
            waste_name = col.replace('accepts_', '').replace('_', ' ').title()
            print(f"   • {waste_name}: {count:,} points")
        elif 'pile' in col.lower() or 'battery' in col.lower():
            # Highlight if battery/pile columns exist but have 0 counts
            waste_name = col.replace('accepts_', '').replace('_', ' ').title()
            print(f"   ⚠️ {waste_name}: {count:,} points (might need manual fix)")

    # Check if we have the right data
    if 'accepts_pile' in df_special_waste.columns:
        pile_count = df_special_waste['accepts_pile'].sum()
        if pile_count == 0 and df_special_waste['accepts_car_battery'].sum() > 0:
            print(f"\n⚠️ WARNING: Small batteries (piles) showing 0 points")
            print("   Try running this SQL in BigQuery to fix:")
            print(f"""
            UPDATE `{PROJECT}.{DATASET}.ecosystem_collection_points_with_coords`
            SET Is_Pile_enabled = 1
            WHERE Is_Car_Battery_enabled = 1
            """)

    # Show sample data
    print(f"\n👀 Sample data (first 3 points):")
    sample_cols = ['point_id', 'name', 'accepts_neon', 'accepts_cartridge',
                   'accepts_battery', 'accepts_car_battery', 'accepts_pile']
    sample_cols = [col for col in sample_cols if col in df_special_waste.columns]
    print(df_special_waste[sample_cols].head(3).to_string(index=False))

except Exception as e:
    print(f"❌ Error retrieving special waste points: {e}")

    # Try a simpler query as last resort
    print("\n🔄 Trying simpler query...")
    try:
        simple_query = f"""
        SELECT
          ID as point_id,
          Name as name,
          Address as address,
          Longitude as longitude,
          Latitude as latitude
        FROM `{PROJECT}.{DATASET}.ecosystem_collection_points_with_coords`
        WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL
        LIMIT 5
        """
        df_simple = client.query(simple_query).to_dataframe()
        print(f"✅ Simple query successful: {len(df_simple)} rows")
        print("Sample:")
        print(df_simple)

        # Check what columns are actually available
        print("\n🔍 To see all available columns, run:")
        print(f"""
        SELECT column_name
        FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'ecosystem_collection_points_with_coords'
        ORDER BY ordinal_position
        """)

    except Exception as e2:
        print(f"❌ Simple query also failed: {e2}")

    df_special_waste = pd.DataFrame()

# Add this at the end of your processing script
print("\n🔍 FINAL DATASET COLUMNS CHECK:")
print("="*50)
print(f"Final DataFrame columns ({len(df_special_waste.columns)} total):")
for col in df_special_waste.columns:
    print(f"  - {col}")

# Check battery-related columns specifically
battery_cols = [col for col in df_special_waste.columns if 'battery' in col.lower() or 'pile' in col.lower()]
print(f"\n🔋 Battery-related columns:")
for col in battery_cols:
    print(f"  - {col}: {df_special_waste[col].sum()} locations accept this type")

# ========================================================
# 6. TEXTILE COLLECTION POINTS (CSV) - PRIORITY 6
# ========================================================

print("\n6. PROCESSING TEXTILE COLLECTION POINTS")
print("-" * 40)

def load_textile_data():
    """Load textile collection points from CSV file"""
    textile_file = "Textile_relais.csv"

    if not os.path.exists(textile_file):
        print(f"⚠️ Textile file not found: {textile_file}")
        return pd.DataFrame()

    try:
        # Try multiple encodings
        encodings = ['latin-1', 'utf-8', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(textile_file, encoding=encoding, on_bad_lines='skip')
                print(f"   Loaded with {encoding} encoding")
                break
            except:
                continue
        else:
            print(f"   Could not read textile file with any encoding")
            return pd.DataFrame()

        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'name' in col_lower or 'nom' in col_lower:
                column_mapping[col] = 'name'
            elif 'adresse' in col_lower or 'address' in col_lower:
                column_mapping[col] = 'address'
            elif 'latitude' in col_lower or 'lat' in col_lower:
                column_mapping[col] = 'latitude'
            elif 'longitude' in col_lower or 'lon' in col_lower or 'long' in col_lower:
                column_mapping[col] = 'longitude'
            elif 'ville' in col_lower or 'city' in col_lower:
                column_mapping[col] = 'city'

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Clean coordinates
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # Filter out invalid coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df[df['latitude'].notna() & df['longitude'].notna()].copy()

        # Add waste type and metadata
        if not df.empty:
            next_id = 80000  # Starting ID for textile points
            df['point_id'] = range(next_id, next_id + len(df))
            df['accepts_textile'] = 1
            df['point_type'] = 'textile_collection'
            df['operator'] = 'Le Relais'
            df['ingestion_date'] = pd.Timestamp.now().date()
            df['schema_version'] = '1.0'
            df['data_source'] = 'Textile_relais.csv'

            # Add other waste types as 0
            waste_columns = ['accepts_food', 'accepts_cardboard', 'accepts_glass',
                           'accepts_metal', 'accepts_paper', 'accepts_plastic',
                           'accepts_wood', 'accepts_vegetation']
            for col in waste_columns:
                df[col] = 0

        print(f"✅ Processed {len(df)} textile collection points")
        return df

    except Exception as e:
        print(f"❌ Error processing textile data: {e}")
        return pd.DataFrame()

df_textile = load_textile_data()

# ========================================================
# 7. PHARMACY/GARAGE/RESSOURCERIE (CSV) - PRIORITY 7
# ========================================================

print("\n7. PROCESSING PHARMACY/GARAGE/RESSOURCERIE POINTS")
print("-" * 40)

def load_mixed_data():
    """Load mixed waste collection points from CSV file"""
    mixed_file = "pharmacies_garages_ressourceries_nantes.csv"

    if not os.path.exists(mixed_file):
        print(f"⚠️ Mixed waste file not found: {mixed_file}")
        return pd.DataFrame()

    try:
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'utf-8-sig', 'cp1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(mixed_file, encoding=encoding, on_bad_lines='skip')
                print(f"   Loaded with {encoding} encoding")
                break
            except:
                continue
        else:
            # Try with semicolon delimiter
            try:
                df = pd.read_csv(mixed_file, sep=';', encoding='latin-1')
                print(f"   Loaded with semicolon delimiter")
            except:
                print(f"   Could not read mixed waste file")
                return pd.DataFrame()

        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'name' in col_lower or 'nom' in col_lower:
                column_mapping[col] = 'name'
            elif 'type' in col_lower or 'categorie' in col_lower:
                column_mapping[col] = 'type'
            elif 'latitude' in col_lower or 'lat' in col_lower:
                column_mapping[col] = 'latitude'
            elif 'longitude' in col_lower or 'lon' in col_lower or 'long' in col_lower:
                column_mapping[col] = 'longitude'
            elif 'adresse' in col_lower or 'address' in col_lower:
                column_mapping[col] = 'address'

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Clean data
        df['name'] = df['name'].fillna('').astype(str).str.strip()
        if 'type' in df.columns:
            df['type'] = df['type'].fillna('').astype(str).str.lower().str.strip()

        # Clean coordinates
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # Filter out invalid coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df[df['latitude'].notna() & df['longitude'].notna()].copy()

        # Add waste types based on type column
        if not df.empty:
            next_id = 90000  # Starting ID for mixed points
            df['point_id'] = range(next_id, next_id + len(df))

            # Initialize all waste columns to 0
            waste_columns = [
                'accepts_food', 'accepts_cardboard', 'accepts_glass',
                'accepts_metal', 'accepts_paper', 'accepts_plastic',
                'accepts_textile', 'accepts_wood', 'accepts_vegetation',
                'accepts_pharmacy', 'accepts_tire', 'accepts_ressourcerie'
            ]

            for col in waste_columns:
                df[col] = 0

            # Set appropriate columns based on type
            if 'type' in df.columns:
                # Pharmacy points
                pharmacy_mask = df['type'].str.contains('pharmacy|pharmacie', case=False, na=False)
                df.loc[pharmacy_mask, 'accepts_pharmacy'] = 1
                df.loc[pharmacy_mask, 'point_type'] = 'pharmacy'

                # Car repair/garage points (tire)
                tire_mask = df['type'].str.contains('car_repair|garage|tire|pneu', case=False, na=False)
                df.loc[tire_mask, 'accepts_tire'] = 1
                df.loc[tire_mask, 'point_type'] = 'car_repair'

                # Ressourcerie points
                ressourcerie_mask = df['type'].str.contains('ressourcerie|recyclerie', case=False, na=False)
                df.loc[ressourcerie_mask, 'accepts_ressourcerie'] = 1
                df.loc[ressourcerie_mask, 'point_type'] = 'ressourcerie'

            # Set default point_type for unclassified
            if 'point_type' not in df.columns:
                df['point_type'] = 'other'

            df['operator'] = 'Various'
            df['ingestion_date'] = pd.Timestamp.now().date()
            df['schema_version'] = '1.0'
            df['data_source'] = 'pharmacies_garages_ressourceries_nantes.csv'

        # Show breakdown
        if 'type' in df.columns:
            unique_types = df['type'].value_counts()
            print("   Breakdown by type:")
            for type_val, count in unique_types.items():
                print(f"   • {type_val}: {count:,} points")

        print(f"✅ Processed {len(df)} mixed waste collection points")
        return df

    except Exception as e:
        print(f"❌ Error processing mixed waste data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

df_mixed = load_mixed_data()

# ========================================================
# 8. COMBINE ALL DATA SOURCES
# ========================================================

print("\n" + "="*60)
print("COMBINING ALL DATA SOURCES")
print("="*60)

# Collect all dataframes
all_dataframes = []

dataframes_info = [
    (df_food, "Food Waste"),
    (df_recycling, "Recycling Centers"),
    (df_glass, "Glass Collection"),
    (df_other_waste, "Other Waste Types"),
    (df_special_waste, "Special Waste"),
    (df_textile, "Textile Collection"),
    (df_mixed, "Mixed Waste (Pharmacy/Garage/Ressourcerie)")
]

for df, name in dataframes_info:
    if not df.empty:
        all_dataframes.append(df)
        print(f"✅ {name}: {len(df):,} points")

if not all_dataframes:
    print("❌ No data to combine!")
    exit()

# Combine all data
combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

print(f"\n📊 DATASET COMPOSITION:")
print(f"   Total points: {len(combined_df):,}")

# ========================================================
# 9. STANDARDIZE AND CLEAN DATA
# ========================================================

print("\n" + "="*60)
print("STANDARDIZING AND CLEANING DATA")
print("="*60)

# Ensure all waste columns are integers (0 or 1)
waste_columns = [col for col in combined_df.columns if col.startswith('accepts_')]
for col in waste_columns:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)
        print(f"   Standardized: {col}")

# Ensure coordinates are numeric
if 'latitude' in combined_df.columns:
    combined_df['latitude'] = pd.to_numeric(combined_df['latitude'], errors='coerce')
if 'longitude' in combined_df.columns:
    combined_df['longitude'] = pd.to_numeric(combined_df['longitude'], errors='coerce')

# Check for missing coordinates
if 'latitude' in combined_df.columns and 'longitude' in combined_df.columns:
    missing_coords = combined_df['latitude'].isna().sum() + combined_df['longitude'].isna().sum()
    if missing_coords > 0:
        print(f"⚠️  {missing_coords} points missing coordinates")
    else:
        print(f"✅ All points have coordinates")

# ========================================================
# 10. CREATE UNIFIED SCHEMA
# ========================================================

print("\n" + "="*60)
print("CREATING UNIFIED SCHEMA")
print("="*60)

# Define core columns (always present)
core_columns = [
    'point_id', 'name', 'address', 'latitude', 'longitude',
    'point_type', 'operator', 'ingestion_date', 'schema_version', 'data_source'
]

# Get all waste columns that exist
existing_waste_cols = [col for col in waste_columns if col in combined_df.columns]

# Get all other columns
other_columns = [col for col in combined_df.columns
                if col not in core_columns and col not in existing_waste_cols]

# Create final column order
final_columns = core_columns + sorted(existing_waste_cols) + sorted(other_columns)

# Ensure all columns exist in dataframe
for col in final_columns:
    if col not in combined_df.columns:
        if col.startswith('accepts_'):
            combined_df[col] = 0
        else:
            combined_df[col] = None

# Reorder columns
combined_df = combined_df[final_columns]

print(f"   Schema created with {len(final_columns)} columns")
print(f"   Waste types tracked: {len(existing_waste_cols)}")

# ========================================================
# 11. SAVE TO CSV (LOCAL BACKUP)
# ========================================================

print("\n" + "="*60)
print("CREATING LOCAL BACKUP")
print("="*60)

output_csv = "trash_collection_points_complete.csv"
try:
    combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ CSV saved: '{output_csv}' ({len(combined_df):,} rows)")

    # Verify CSV
    csv_check = pd.read_csv(output_csv, nrows=5)
    print(f"   CSV verification: {len(csv_check)} rows read successfully")

    # Show sample
    print(f"\n📋 CSV SAMPLE (first 3 rows):")
    sample = combined_df.head(3)
    for i, row in sample.iterrows():
        point_id = row.get('point_id', 'N/A')
        name = str(row.get('name', 'Unnamed'))[:40]
        point_type = row.get('point_type', 'Unknown')

        # Find accepted waste types
        accepted = []
        for col in existing_waste_cols:
            if row.get(col, 0) == 1:
                waste_name = col.replace('accepts_', '').replace('_', ' ').title()
                accepted.append(waste_name)

        accepted_str = ', '.join(accepted[:2]) + ('...' if len(accepted) > 2 else '')
        print(f"   ID {point_id}: {name}... [{point_type}] → {accepted_str}")

except Exception as e:
    print(f"❌ Error saving CSV: {e}")

# ========================================================
# 12. UPLOAD TO BIGQUERY
# ========================================================

print("\n" + "="*60)
print("UPLOADING TO BIGQUERY")
print("="*60)

table_id = f"{PROJECT}.{DATASET}.{TARGET_TABLE}"

# Check if table exists, create backup if it does
try:
    existing_table = client.get_table(table_id)
    print(f"ℹ️ Table {table_id} already exists ({existing_table.num_rows:,} rows)")

    # Create backup
    backup_table = f"{PROJECT}.{DATASET}.{TARGET_TABLE}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backup_query = f"""
    CREATE OR REPLACE TABLE `{backup_table}` AS
    SELECT * FROM `{table_id}`
    """

    client.query(backup_query).result()
    print(f"✅ Backup created: {backup_table}")

except NotFound:
    print(f"ℹ️ Table {table_id} does not exist, will create new")

# Upload data to BigQuery
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # Replace entire table
    autodetect=True,                     # Auto-detect schema
    max_bad_records=10                   # Allow some bad records
)

try:
    print(f"📤 Uploading {len(combined_df):,} rows to BigQuery...")
    job = client.load_table_from_dataframe(combined_df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete

    # Verify upload
    table = client.get_table(table_id)
    print(f"✅ BigQuery table created/updated: {table_id}")
    print(f"   • Rows: {table.num_rows:,}")
    print(f"   • Size: {table.num_bytes / (1024*1024):.2f} MB")

except Exception as e:
    print(f"❌ BigQuery upload failed: {e}")

    # Try alternative method
    try:
        print("🔄 Trying alternative upload method...")
        import pandas_gbq
        pandas_gbq.to_gbq(
            combined_df,
            destination_table=table_id,
            project_id=PROJECT,
            if_exists='replace',
            progress_bar=True
        )
        print("✅ Upload successful via pandas_gbq!")
    except Exception as e2:
        print(f"❌ Alternative also failed: {e2}")
        print("\n📋 Manual upload instructions:")
        print(f"   1. Go to BigQuery Console: https://console.cloud.google.com/bigquery")
        print(f"   2. Select project: {PROJECT}")
        print(f"   3. Select dataset: {DATASET}")
        print(f"   4. Create table: {TARGET_TABLE}")
        print(f"   5. Upload file: {output_csv}")
        print(f"   6. Enable schema autodetection")

# ========================================================
# 13. VERIFICATION AND STATISTICS
# ========================================================

print("\n" + "="*60)
print("GENERATING STATISTICS AND VERIFICATION")
print("="*60)

try:
    # Run verification query
    verify_query = f"""
    SELECT
      COUNT(*) as total_points,
      SUM(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 ELSE 0 END) as points_with_coords,
      {', '.join([f'SUM({col}) as {col}_count' for col in existing_waste_cols[:10]])}
    FROM `{table_id}`
    """

    result = client.query(verify_query).to_dataframe().iloc[0]

    print(f"📊 VERIFICATION RESULTS:")
    print(f"   • Total points: {result['total_points']:,}")
    print(f"   • With coordinates: {result['points_with_coords']:,}")

    print(f"\n   • Waste type coverage (top 10):")
    waste_counts = []
    for col in existing_waste_cols[:10]:
        count = int(result.get(f'{col}_count', 0))
        if count > 0:
            waste_name = col.replace('accepts_', '').replace('_', ' ').title()
            percentage = (count / result['total_points']) * 100
            waste_counts.append((waste_name, count, percentage))

    # Sort by count descending
    waste_counts.sort(key=lambda x: x[1], reverse=True)

    for waste_name, count, percentage in waste_counts:
        print(f"     - {waste_name:18s}: {count:6,d} ({percentage:5.1f}%)")

    if len(existing_waste_cols) > 10:
        print(f"     ... and {len(existing_waste_cols) - 10} more waste types")

    # Point type distribution
    type_query = f"""
    SELECT
      point_type,
      COUNT(*) as count,
      ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
    FROM `{table_id}`
    WHERE point_type IS NOT NULL
    GROUP BY point_type
    ORDER BY count DESC
    LIMIT 10
    """

    type_results = client.query(type_query).to_dataframe()

    print(f"\n   • Point type distribution (top 10):")
    for _, row in type_results.iterrows():
        print(f"     - {row['point_type']:25s}: {row['count']:6,d} ({row['percentage']:5.1f}%)")

except Exception as e:
    print(f"⚠️ Verification query failed: {e}")

# ========================================================
# 14. FINAL SUMMARY
# ========================================================

print("\n" + "="*60)
print("DATASET CREATION COMPLETE!")
print("="*60)

print(f"\n🎯 SUMMARY:")
print(f"   • Total collection points: {len(combined_df):,}")
print(f"   • Data sources integrated: {len(all_dataframes)}")
print(f"   • Waste types tracked: {len(existing_waste_cols)}")
print(f"   • BigQuery table: {table_id}")
print(f"   • Local backup: {output_csv}")

# Data quality check
print(f"\n🔍 DATA QUALITY CHECK:")
print(f"   • All waste columns standardized to 0/1")
print(f"   • Coordinates validated and cleaned")

if 'latitude' in combined_df.columns and 'longitude' in combined_df.columns:
    valid_coords = combined_df['latitude'].notna().sum()
    coord_percentage = (valid_coords / len(combined_df)) * 100
    print(f"   • Coordinate completeness: {coord_percentage:.1f}%")

# Recommendations
print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Verify data in BigQuery Console")
print(f"   2. Test queries on the new dataset")
print(f"   3. Schedule regular updates with new data")
print(f"   4. Consider adding geospatial indexes for better performance")

print(f"\n" + "="*60)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("="*60)

# Save final statistics
stats = {
    'timestamp': datetime.now().isoformat(),
    'total_points': len(combined_df),
    'data_sources': len(all_dataframes),
    'waste_types': len(existing_waste_cols),
    'table_name': table_id,
    'local_backup': output_csv,
    'schema_version': '1.0'
}

stats_file = "ingestion_statistics.json"
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n📊 Statistics saved to: {stats_file}")
