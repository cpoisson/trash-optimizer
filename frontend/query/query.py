from dotenv import load_dotenv
import os
import pandas as pd
from google.cloud import bigquery
from  .params import category_mapping

def get_loc(list_trash = None):
    if list_trash is None:
        list_trash = ["Food Organics"]
    load_dotenv()
    print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    GCP_PROJECT = os.getenv("GCP_PROJECT")
    GCP_DATASET = os.getenv("GCP_DATASET")
    bigquery_client = bigquery.Client(project=GCP_PROJECT)
    dfs = []

    for category in list_trash:
        if category not in category_mapping:
            print(f"Category '{category}' not in mapping, skipping.")
            continue

        query = f"""
        SELECT
            Name,
            Address,
            Longitude,
            Latitude,
            '{category}' AS Trash_class
        FROM `{GCP_PROJECT}.{GCP_DATASET}.trash_collection_points`
        WHERE {category_mapping[category]}
        """

        df = bigquery_client.query(query).to_dataframe()
        dfs.append(df)

    if dfs:
        # Concatène tous les DataFrames et supprime les doublons
        result_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        return result_df
    else:
        return pd.DataFrame(columns=["Name", "Address", "Longitude", "Latitude", "Trash_class"])
