#!/usr/bin/env python3
"""
Fix Is_Pile_enabled Column

This script updates the ecosystem_collection_points_with_coords table
to ensure that all locations accepting car batteries also accept small batteries (piles).
"""

from google.cloud import bigquery
from dotenv import load_dotenv
import os
import bigquery_utils as bq_utils

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
PROJECT = os.getenv('GCP_PROJECT')
DATASET = os.getenv('GCP_DATASET')
TABLE = "ecosystem_collection_points_with_coords"

# Validate required environment variables
if not all([CREDENTIALS_PATH, PROJECT, DATASET]):
    raise ValueError(
        "Missing required environment variables. Please check your .env file.\n"
        "Required: GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT, GCP_DATASET\n"
        "Copy .env.template to .env and fill in your values."
    )

# Set GCP credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# Initialize client
table_id = f"{PROJECT}.{DATASET}.{TABLE}"
client = bq_utils.get_bigquery_client(PROJECT)

print("🔧 FIXING Is_Pile_enabled COLUMN")
print("="*50)

# Step 1: Check current state
print("\n📊 CURRENT STATE (Before fix):")
check_query = f"""
SELECT
  COUNT(*) as total_points,
  SUM(Is_Car_Battery_enabled) as car_battery_points,
  SUM(Is_Pile_enabled) as pile_points,
  SUM(CASE WHEN Is_Car_Battery_enabled = 1 AND Is_Pile_enabled = 0 THEN 1 ELSE 0 END) as need_fixing
FROM `{table_id}`
"""

try:
    result = client.query(check_query).to_dataframe()
    print(f"Total points: {result.iloc[0]['total_points']}")
    print(f"Car battery points: {result.iloc[0]['car_battery_points']}")
    print(f"Pile points: {result.iloc[0]['pile_points']}")
    print(f"Points needing fix: {result.iloc[0]['need_fixing']}")
except Exception as e:
    print(f"Error checking: {e}")

# Step 2: Run the update
print("\n🔄 RUNNING UPDATE...")
update_query = f"""
UPDATE `{table_id}`
SET Is_Pile_enabled = 1
WHERE Is_Car_Battery_enabled = 1
"""

try:
    # Run the update
    job = client.query(update_query)
    job.result()  # Wait for job to complete

    print(f"✅ Update completed successfully!")
    print(f"   Job ID: {job.job_id}")

    # Check how many rows were updated
    if hasattr(job, 'num_dml_affected_rows'):
        print(f"   Rows updated: {job.num_dml_affected_rows}")

except Exception as e:
    print(f"❌ Update failed: {e}")

# Step 3: Verify the fix
print("\n✅ VERIFICATION (After fix):")
verify_query = f"""
SELECT
  COUNT(*) as total_points,
  SUM(Is_Car_Battery_enabled) as car_battery_points,
  SUM(Is_Pile_enabled) as pile_points,
  SUM(CASE WHEN Is_Car_Battery_enabled = 1 AND Is_Pile_enabled = 1 THEN 1 ELSE 0 END) as both_types
FROM `{table_id}`
"""

try:
    result = client.query(verify_query).to_dataframe()
    print(f"Total points: {result.iloc[0]['total_points']}")
    print(f"Car battery points: {result.iloc[0]['car_battery_points']}")
    print(f"Pile points: {result.iloc[0]['pile_points']}")
    print(f"Points with both types: {result.iloc[0]['both_types']}")

    if result.iloc[0]['car_battery_points'] == result.iloc[0]['pile_points']:
        print("\n🎉 SUCCESS! All car battery locations now accept small batteries too!")
    else:
        print("\n⚠️ WARNING: Counts don't match. Something went wrong.")

except Exception as e:
    print(f"Error verifying: {e}")

print("\n" + "="*50)
print("🎯 FIX COMPLETE!")
