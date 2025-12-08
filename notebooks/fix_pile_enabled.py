from google.cloud import bigquery

# Set credentials
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/dariaserbichenko/code/DariaSerb/key-gcp/trash-optimizer-479913-91e59ecc96c9.json"

# Initialize client
PROJECT = "trash-optimizer-479913"
DATASET = "nantes"
TABLE = "ecosystem_collection_points_with_coords"
table_id = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client(project=PROJECT)

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
