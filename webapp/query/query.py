from dotenv import load_dotenv
import os
import pandas as pd
from google.cloud import bigquery
from  .params import category_mapping

def get_loc(list_trash = None):
    """
    Fetch drop-off locations from BigQuery for specified trash categories.
    Optimized to use single query with OR conditions instead of multiple queries.
    """
    if list_trash is None:
        list_trash = ["food_organics"]
    load_dotenv()
    print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    GCP_PROJECT = os.getenv("GCP_PROJECT")
    GCP_DATASET = os.getenv("GCP_DATASET")
    bigquery_client = bigquery.Client(project=GCP_PROJECT)

    # Filter out unmapped categories
    valid_categories = [cat for cat in list_trash if cat in category_mapping]

    if not valid_categories:
        print("No valid categories found in mapping.")
        return pd.DataFrame(columns=["Name", "Address", "Longitude", "Latitude", "Trash_class"])

    # Build single query with UNION ALL for better performance
    subqueries = []
    for category in valid_categories:
        subquery = f"""
        SELECT
            Name,
            Address,
            Longitude,
            Latitude,
            '{category}' AS Trash_class
        FROM `{GCP_PROJECT}.{GCP_DATASET}.trash_collection_points`
        WHERE {category_mapping[category]}
        AND Longitude IS NOT NULL
        AND Latitude IS NOT NULL
        """
        subqueries.append(subquery)

    # Combine all subqueries with UNION ALL
    query = " UNION ALL ".join(subqueries)

    try:
        df = bigquery_client.query(query).to_dataframe()
        return df.drop_duplicates()
    except Exception as e:
        print(f"BigQuery error: {e}")
        return pd.DataFrame(columns=["Name", "Address", "Longitude", "Latitude", "Trash_class"])
