"""
Create Trash Collection Points Database
========================================
Unified script to create and maintain trash collection points database
from multiple sources and upload to BigQuery.
"""

# IMPORT LIBRARIES
import os
import re
import pandas as pd
from google.cloud import bigquery

# SET CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/dariaserbichenko/code/DariaSerb/key-gcp/trash-optimizer-479913-91e59ecc96c9.json"

PROJECT = "trash-optimizer-479913"
DATASET = "nantes"
client = bigquery.Client(project=PROJECT)

print("="*60)
print("Creating optimized trash collection points table")
print("="*60)

# ========================================================
# 1. QUERY FOR ALIMENTARY GARBAGE (FOOD WASTE)
# ========================================================

print("="*60)
print("\n1. Querying alimentary garbage (food waste)")
print("="*60)

query1 = f"""
SELECT
  ROW_NUMBER() OVER () as ID,
  CONCAT('Food Waste - ', COALESCE(commune, 'Nantes')) as Name,
  COALESCE(adresse, 'Address not specified') as Address,
  lon as Longitude,
  lat as Latitude,
  0 as Is_Cardboard_enabled,
  1 as Is_Food_enabled,
  0 as Is_Glass_enabled,
  0 as Is_Metal_enabled,
  0 as Is_Paper_enabled,
  0 as Is_Wood_enabled,
  0 as Is_Plastic_enabled,
  0 as Is_Textile_enabled,
  0 as Is_Vegetation_enabled,
  0 as Is_Neon_enabled,
  0 as Is_Cartridge_enabled,
  0 as Is_Lamp_Light_enabled,
  0 as Is_Pile_enabled,
  0 as Is_Battery_enabled,
  0 as Is_Car_Battery_enabled,
  0 as Is_Miscellanous_Trash_enabled,
  0 as Is_Pharmacy_enabled,
  0 as Is_Tire_enabled,
  0 as Is_Ressourcerie_enabled
FROM `{PROJECT}.{DATASET}.alimentary_garbage_clean`
WHERE lat IS NOT NULL AND lon IS NOT NULL
"""

try:
    df1 = client.query(query1).to_dataframe()
    print(f"Retrieved {len(df1):,} food waste locations")
except Exception as e:
    print(f"Error: {e}")
    df1 = pd.DataFrame()

# ========================================================
# 2. QUERY FOR ECOPOINTS WITH ACTUAL COLUMNS
# ========================================================

print("="*60)
print("\n2. Querying ecopoints with actual columns")
print("="*60)

query2 = f"""
SELECT
  ROW_NUMBER() OVER () + 10000 as ID,
  CONCAT('Recycling Center - ', COALESCE(nom, commune, 'Ecopoint')) as Name,
  COALESCE(adresse, 'Address not specified') as Address,
  lon as Longitude,
  lat as Latitude,
  CASE WHEN UPPER(carton) = 'OUI' THEN 1 ELSE 0 END as Is_Cardboard_enabled,
  0 as Is_Food_enabled,
  CASE WHEN UPPER(verre) = 'OUI' THEN 1 ELSE 0 END as Is_Glass_enabled,
  CASE WHEN UPPER(ferraille) = 'OUI' THEN 1 ELSE 0 END as Is_Metal_enabled,
  CASE WHEN UPPER(papier) = 'OUI' THEN 1 ELSE 0 END as Is_Paper_enabled,
  CASE WHEN UPPER(bois) = 'OUI' THEN 1 ELSE 0 END as Is_Wood_enabled,
  0 as Is_Plastic_enabled,
  CASE WHEN UPPER(textile) = 'OUI' THEN 1 ELSE 0 END as Is_Textile_enabled,
  CASE WHEN UPPER(dechet_vert) = 'OUI' THEN 1 ELSE 0 END as Is_Vegetation_enabled,
  CASE WHEN UPPER(neon) = 'OUI' THEN 1 ELSE 0 END as Is_Neon_enabled,
  CASE WHEN UPPER(cartouche) = 'OUI' THEN 1 ELSE 0 END as Is_Cartridge_enabled,
  0 as Is_Lamp_Light_enabled,
  CASE WHEN UPPER(pile) = 'OUI' THEN 1 ELSE 0 END as Is_Pile_enabled,
  CASE WHEN UPPER(batterie) = 'OUI' THEN 1 ELSE 0 END as Is_Car_Battery_enabled,
  0 as Is_Miscellanous_Trash_enabled,
  0 as Is_Pharmacy_enabled,
  CASE WHEN UPPER(pneus) = 'OUI' THEN 1 ELSE 0 END as Is_Tire_enabled,
  0 as Is_Ressourcerie_enabled
FROM `{PROJECT}.{DATASET}.ecopoints`
WHERE lat IS NOT NULL AND lon IS NOT NULL
"""

try:
    df2 = client.query(query2).to_dataframe()
    print(f"Retrieved {len(df2)} recycling centers with actual waste types")

    waste_cols = [col for col in df2.columns if col.startswith('Is_')]
    print(f"- Waste acceptance in recycling centers:")
    for col in waste_cols:
        count = df2[col].sum()
        if count > 0:
            waste_name = col.replace('Is_', '').replace('_enabled', '').replace('_', ' ').title()
            print(f"   {waste_name}: {count}/{len(df2)} locations")

except Exception as e:
    print(f"Error: {e}")
    df2 = pd.DataFrame()

# ========================================================
# 3. QUERY FOR GLASS COLLECTION POINTS
# ========================================================

print("="*60)
print("\n3. Querying glass collection columns (Verre only)")
print("="*60)

query3 = f"""
SELECT
  ROW_NUMBER() OVER () + 30000 as ID,
  CONCAT(
    'Drop-off points - ',
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
      'Glass Collection'
    ),
    CASE
      WHEN commune IS NOT NULL THEN CONCAT(' - ', commune)
      ELSE ' - Nantes'
    END
  ) as Name,
  COALESCE(adresse, 'Nantes Métropole') as Address,
  lat as Latitude,
  lon as Longitude,
  0 as Is_Cardboard_enabled,
  0 as Is_Food_enabled,
  1 as Is_Glass_enabled,
  0 as Is_Metal_enabled,
  0 as Is_Paper_enabled,
  0 as Is_Wood_enabled,
  0 as Is_Plastic_enabled,
  0 as Is_Textile_enabled,
  0 as Is_Vegetation_enabled,
  0 as Is_Neon_enabled,
  0 as Is_Cartridge_enabled,
  0 as Is_Lamp_Light_enabled,
  0 as Is_Pile_enabled,
  0 as Is_Battery_enabled,
  0 as Is_Car_Battery_enabled,
  0 as Is_Miscellanous_Trash_enabled,
  0 as Is_Pharmacy_enabled,
  0 as Is_Tire_enabled,
  0 as Is_Ressourcerie_enabled
FROM `{PROJECT}.{DATASET}.location_dropoff_points_nantes`
WHERE
  lat IS NOT NULL
  AND lon IS NOT NULL
  AND LOWER(TRIM(type_dechet)) = 'verre'
"""

try:
    df3 = client.query(query3).to_dataframe()
    print(f"Retrieved {len(df3):,} glass collection columns")
except Exception as e:
    print(f"Error: {e}")
    df3 = pd.DataFrame()

# ========================================================
# 4. QUERY FOR NON-GLASS WASTE TYPES
# ========================================================

print("="*60)
print("\n4. Querying non-glass waste columns with waste type names")
print("="*60)

query4 = f"""
SELECT
  ROW_NUMBER() OVER () +
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 40000
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 50000
    WHEN LOWER(TRIM(type_dechet)) LIKE '%ordure ménagère%' THEN 60000
    ELSE 70000
  END as ID,
  CONCAT(
    CASE
      WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 'Paper/Cardboard'
      WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 'Recyclable Waste'
      WHEN LOWER(TRIM(type_dechet)) LIKE '%ordure ménagère%' THEN 'Household Waste'
      ELSE INITCAP(type_dechet)
    END,
    ' Drop-off Point - ',
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
  ) as Name,
  COALESCE(adresse, 'Nantes Métropole') as Address,
  lat as Latitude,
  lon as Longitude,
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 1
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as Is_Cardboard_enabled,
  0 as Is_Food_enabled,
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%verre%' THEN 1
    ELSE 0
  END as Is_Glass_enabled,
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as Is_Metal_enabled,
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%papier%carton%' THEN 1
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as Is_Paper_enabled,
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%déchet recyclable%' THEN 1
    ELSE 0
  END as Is_Plastic_enabled,
  CASE
    WHEN LOWER(TRIM(type_dechet)) LIKE '%ordure ménagère%' THEN 1
    ELSE 0
  END as Is_Miscellanous_Trash_enabled,
  0 as Is_Textile_enabled,
  0 as Is_Vegetation_enabled,
  0 as Is_Neon_enabled,
  0 as Is_Cartridge_enabled,
  0 as Is_Lamp_Light_enabled,
  type_dechet as Original_Waste_Type,
  type_colonne as Original_Column_Type,
  commune as Commune
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
    df4 = client.query(query4).to_dataframe()
    print(f"Retrieved {len(df4):,} non-glass waste columns")
except Exception as e:
    print(f"Error: {e}")
    df4 = pd.DataFrame()

# ========================================================
# 5. QUERY FOR ECOSYSTEM POINTS
# ========================================================

print("="*60)
print("\n5. Querying ecosystem collection points")
print("="*60)

query5 = f"""
SELECT
  ID,
  Name,
  Address,
  Longitude,
  Latitude,
  Is_Neon_enabled,
  Is_Cartridge_enabled,
  Is_Lamp_Light_enabled,
  Is_Battery_enabled,
  Is_Car_Battery_enabled,
  Is_Pile_enabled
FROM `{PROJECT}.{DATASET}.ecosystem_collection_points_with_coords`
WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL
"""

try:
    df5 = client.query(query5).to_dataframe()
    print(f"Retrieved {len(df5):,} Ecosystem collection points")
except Exception as e:
    print(f"Error retrieving ecosystem points: {e}")
    df5 = pd.DataFrame()

# ========================================================
# 6. LOAD TEXTILE DATA FROM CSV
# ========================================================

print("\n6. LOADING TEXTILE DATA")
print("-" * 40)

textile_file = "Textile_relais.csv"
textile_df = pd.DataFrame()

if os.path.exists(textile_file):
    try:
        encodings = ['latin-1', 'utf-8', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                textile_df = pd.read_csv(textile_file, encoding=encoding, on_bad_lines='skip')
                print(f"   Loaded with {encoding} encoding")
                break
            except:
                continue

        if not textile_df.empty:
            # Standardize column names
            column_mapping = {}
            for col in textile_df.columns:
                col_lower = str(col).lower().strip()
                if 'name' in col_lower or 'nom' in col_lower:
                    column_mapping[col] = 'Name'
                elif 'adresse' in col_lower or 'address' in col_lower:
                    column_mapping[col] = 'Address'
                elif 'latitude' in col_lower or 'lat' in col_lower:
                    column_mapping[col] = 'Latitude'
                elif 'longitude' in col_lower or 'lon' in col_lower:
                    column_mapping[col] = 'Longitude'

            if column_mapping:
                textile_df = textile_df.rename(columns=column_mapping)

            # Clean coordinates
            if 'Latitude' in textile_df.columns:
                textile_df['Latitude'] = pd.to_numeric(textile_df['Latitude'], errors='coerce')
            if 'Longitude' in textile_df.columns:
                textile_df['Longitude'] = pd.to_numeric(textile_df['Longitude'], errors='coerce')

            # Filter valid coordinates
            textile_df = textile_df[textile_df['Latitude'].notna() & textile_df['Longitude'].notna()].copy()

            # Add waste type capabilities
            next_id = 80000
            textile_df['ID'] = range(next_id, next_id + len(textile_df))
            textile_df['Is_Textile_enabled'] = 1

            # Initialize other waste types
            for col in ['Is_Cardboard_enabled', 'Is_Food_enabled', 'Is_Glass_enabled',
                       'Is_Metal_enabled', 'Is_Paper_enabled', 'Is_Plastic_enabled',
                       'Is_Wood_enabled', 'Is_Vegetation_enabled', 'Is_Neon_enabled',
                       'Is_Cartridge_enabled', 'Is_Lamp_Light_enabled',
                       'Is_Miscellanous_Trash_enabled', 'Is_Pharmacy_enabled',
                       'Is_Tire_enabled', 'Is_Ressourcerie_enabled']:
                textile_df[col] = 0

            print(f"✅ Processed {len(textile_df)} textile collection points")
        else:
            print("   Could not load textile file")
            textile_df = pd.DataFrame()

    except Exception as e:
        print(f"❌ Error loading textile data: {e}")
        textile_df = pd.DataFrame()
else:
    print(f"⚠️ Textile file not found: {textile_file}")

# ========================================================
# 7. LOAD MIXED DATA FROM CSV
# ========================================================

print("\n7. LOADING PHARMACY/GARAGE/RESSOURCERIE DATA")
print("-" * 40)

mixed_file = "pharmacies_garages_ressourceries_nantes.csv"
mixed_df = pd.DataFrame()

if os.path.exists(mixed_file):
    try:
        encodings = ['utf-8', 'latin-1', 'utf-8-sig', 'cp1252']
        for encoding in encodings:
            try:
                mixed_df = pd.read_csv(mixed_file, encoding=encoding, on_bad_lines='skip')
                print(f"   Loaded with {encoding} encoding")
                break
            except:
                continue

        if mixed_df is None:
            try:
                mixed_df = pd.read_csv(mixed_file, sep=';', encoding='latin-1')
                print("   Loaded with ';' delimiter")
            except:
                raise Exception("Could not read mixed data file")

        if not mixed_df.empty:
            # Standardize column names
            column_mapping = {}
            for col in mixed_df.columns:
                col_lower = str(col).lower().strip()
                if 'name' in col_lower or 'nom' in col_lower:
                    column_mapping[col] = 'Name'
                elif 'type' in col_lower or 'categorie' in col_lower:
                    column_mapping[col] = 'Type'
                elif 'latitude' in col_lower or 'lat' in col_lower:
                    column_mapping[col] = 'Latitude'
                elif 'longitude' in col_lower or 'lon' in col_lower:
                    column_mapping[col] = 'Longitude'
                elif 'adresse' in col_lower or 'address' in col_lower:
                    column_mapping[col] = 'Address'

            if column_mapping:
                mixed_df = mixed_df.rename(columns=column_mapping)

            # Clean data
            mixed_df['Name'] = mixed_df['Name'].fillna('').astype(str).str.strip()
            if 'Type' in mixed_df.columns:
                mixed_df['Type'] = mixed_df['Type'].fillna('').astype(str).str.lower().str.strip()

            # Clean coordinates
            if 'Latitude' in mixed_df.columns:
                mixed_df['Latitude'] = pd.to_numeric(mixed_df['Latitude'], errors='coerce')
            if 'Longitude' in mixed_df.columns:
                mixed_df['Longitude'] = pd.to_numeric(mixed_df['Longitude'], errors='coerce')

            # Filter valid coordinates
            mixed_df = mixed_df[mixed_df['Latitude'].notna() & mixed_df['Longitude'].notna()].copy()

            # Add waste types based on type
            next_id = 90000
            mixed_df['ID'] = range(next_id, next_id + len(mixed_df))

            # Initialize all waste columns
            for col in ['Is_Cardboard_enabled', 'Is_Food_enabled', 'Is_Glass_enabled',
                       'Is_Metal_enabled', 'Is_Paper_enabled', 'Is_Plastic_enabled',
                       'Is_Textile_enabled', 'Is_Wood_enabled', 'Is_Vegetation_enabled',
                       'Is_Neon_enabled', 'Is_Cartridge_enabled', 'Is_Lamp_Light_enabled',
                       'Is_Miscellanous_Trash_enabled', 'Is_Pharmacy_enabled',
                       'Is_Tire_enabled', 'Is_Ressourcerie_enabled']:
                mixed_df[col] = 0

            # Classify by type
            if 'Type' in mixed_df.columns:
                pharmacy_mask = mixed_df['Type'].str.contains('pharmacy|pharmacie', case=False, na=False)
                mixed_df.loc[pharmacy_mask, 'Is_Pharmacy_enabled'] = 1

                tire_mask = mixed_df['Type'].str.contains('car_repair|garage|tire|pneu', case=False, na=False)
                mixed_df.loc[tire_mask, 'Is_Tire_enabled'] = 1

                ressourcerie_mask = mixed_df['Type'].str.contains('ressourcerie|recyclerie', case=False, na=False)
                mixed_df.loc[ressourcerie_mask, 'Is_Ressourcerie_enabled'] = 1

            print(f"✅ Processed {len(mixed_df)} mixed collection points")
        else:
            print("   Could not load mixed file")
            mixed_df = pd.DataFrame()

    except Exception as e:
        print(f"❌ Error loading mixed data: {e}")
        mixed_df = pd.DataFrame()
else:
    print(f"⚠️ Mixed file not found: {mixed_file}")

# ========================================================
# 8. COMBINE ALL DATASETS
# ========================================================

print("\n" + "="*60)
print("COMBINING ALL DATASETS")
print("="*60)

all_dataframes = []

for df, name in [(df1, "Food Waste"), (df2, "Recycling Centers"),
                 (df3, "Glass Collection"), (df4, "Other Waste Types"),
                 (df5, "Ecosystem Points"), (textile_df, "Textile Collection"),
                 (mixed_df, "Mixed Collection")]:
    if not df.empty:
        all_dataframes.append(df)
        print(f"✅ {name}: {len(df):,} points")

if not all_dataframes:
    print("❌ No data to combine!")
    exit()

# Combine all data
combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
print(f"\n📊 Combined dataset: {len(combined_df):,} points")

# Reset ID to be sequential
combined_df['ID'] = range(1, len(combined_df) + 1)

# Define final columns
final_columns = [
    'ID', 'Name', 'Address', 'Longitude', 'Latitude',
    'Is_Cardboard_enabled', 'Is_Food_enabled', 'Is_Glass_enabled',
    'Is_Metal_enabled', 'Is_Paper_enabled', 'Is_Wood_enabled', 'Is_Plastic_enabled',
    'Is_Textile_enabled', 'Is_Vegetation_enabled', 'Is_Neon_enabled',
    'Is_Cartridge_enabled', 'Is_Lamp_Light_enabled', 'Is_Pile_enabled',
    'Is_Battery_enabled', 'Is_Car_Battery_enabled',
    'Is_Miscellanous_Trash_enabled', 'Is_Pharmacy_enabled',
    'Is_Tire_enabled', 'Is_Ressourcerie_enabled'
]

# Ensure all columns exist
for col in final_columns:
    if col not in combined_df.columns:
        if col.startswith('Is_'):
            combined_df[col] = 0
        else:
            combined_df[col] = None

# Convert waste columns to integers
for col in combined_df.columns:
    if col.startswith('Is_'):
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)

# Reorder columns
available_columns = [col for col in final_columns if col in combined_df.columns]
combined_df = combined_df[available_columns]

print(f"\n✅ Final dataset: {len(combined_df):,} points")

# ========================================================
# 9. SAVE TO CSV
# ========================================================

print("\n" + "="*60)
print("SAVING TO CSV")
print("="*60)

output_csv = "trash_collection_points_complete.csv"
combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✅ CSV saved: '{output_csv}' ({len(combined_df):,} rows)")

# ========================================================
# 10. UPLOAD TO BIGQUERY
# ========================================================

print("\n" + "="*60)
print("UPLOADING TO BIGQUERY")
print("="*60)

table_id = f"{PROJECT}.{DATASET}.trash_collection_points"

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
    autodetect=True,
    max_bad_records=100
)

try:
    job = client.load_table_from_dataframe(combined_df, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"✅ BigQuery table created: {table_id}")
    print(f"   Rows: {table.num_rows:,}")
    print(f"   Size: {table.num_bytes / (1024*1024):.2f} MB")

    # Verification query
    verify_query = f"""
    SELECT
      COUNT(*) as total_points,
      SUM(Is_Textile_enabled) as textile_points,
      SUM(Is_Pharmacy_enabled) as pharmacy_points,
      SUM(Is_Tire_enabled) as tire_points,
      SUM(Is_Ressourcerie_enabled) as ressourcerie_points,
      SUM(Is_Glass_enabled) as glass_points,
      SUM(Is_Food_enabled) as food_points
    FROM `{table_id}`
    """

    result = client.query(verify_query).to_dataframe().iloc[0]
    print(f"\n🔍 VERIFICATION:")
    print(f"  • Total points: {result['total_points']:,}")
    print(f"  • Textile: {result['textile_points']:,}")
    print(f"  • Pharmacy: {result['pharmacy_points']:,}")
    print(f"  • Tire: {result['tire_points']:,}")
    print(f"  • Ressourcerie: {result['ressourcerie_points']:,}")
    print(f"  • Glass: {result['glass_points']:,}")
    print(f"  • Food: {result['food_points']:,}")

except Exception as e:
    print(f"❌ BigQuery upload failed: {e}")

print("\n" + "="*60)
print("PROCESS COMPLETE!")
print("="*60)
