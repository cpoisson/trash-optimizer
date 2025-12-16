"""
Shared BigQuery utilities for trash optimizer data scripts.

Provides common functions for:
- BigQuery client initialization
- DataFrame upload with error handling
- DataFrame cleaning and preparation
- Coordinate extraction and validation
"""

import io
import os
from typing import Optional, Dict, Any
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound


def get_bigquery_client(project: str) -> bigquery.Client:
    """
    Initialize BigQuery client with error handling.

    Parameters:
    -----------
    project : str
        GCP project ID

    Returns:
    --------
    bigquery.Client
        Initialized BigQuery client
    """
    try:
        client = bigquery.Client(project=project)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize BigQuery client: {e}")


def ensure_dataset_exists(client: bigquery.Client, project: str, dataset: str, location: str = "EU") -> None:
    """
    Ensure BigQuery dataset exists, create if not.

    Parameters:
    -----------
    client : bigquery.Client
        BigQuery client
    project : str
        GCP project ID
    dataset : str
        Dataset name
    location : str, optional
        Dataset location (default: "EU")
    """
    dataset_ref = f"{project}.{dataset}"
    try:
        client.get_dataset(dataset_ref)
        print(f"✅ Dataset '{dataset}' exists")
    except NotFound:
        print(f"📁 Creating dataset '{dataset}'...")
        dataset_obj = bigquery.Dataset(dataset_ref)
        dataset_obj.location = location
        client.create_dataset(dataset_obj, timeout=30)
        print(f"✅ Dataset created in {location}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names for BigQuery compatibility.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    df_clean.columns = df_clean.columns.str.lower()
    return df_clean


def convert_complex_types_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert lists, dicts, and tuples to strings for BigQuery compatibility.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with complex types converted to strings
    """
    df_clean = df.copy()
    conversions = 0

    for col in df_clean.columns:
        if df_clean[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
            df_clean[col] = df_clean[col].astype(str)
            conversions += 1
            print(f"   Converted {col} to string")

    if conversions > 0:
        print(f"   Total conversions: {conversions}")

    return df_clean


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values appropriately based on column dtype.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled
    """
    df_clean = df.copy()
    nan_count = df_clean.isna().sum().sum()

    if nan_count > 0:
        print(f"   Found {nan_count} NaN values")
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('')

    return df_clean


def prepare_dataframe_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for BigQuery upload.

    Applies:
    - Column name cleaning
    - Complex type conversion
    - Missing value handling

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame ready for BigQuery
    """
    print("🧹 Preparing DataFrame for BigQuery...")

    df_clean = clean_column_names(df)
    df_clean = convert_complex_types_to_string(df_clean)
    df_clean = fill_missing_values(df_clean)

    print(f"   Final shape: {df_clean.shape}")
    return df_clean


def upload_dataframe_to_bigquery(
    df: pd.DataFrame,
    table_name: str,
    project: str,
    dataset: str,
    client: Optional[bigquery.Client] = None,
    write_disposition: str = "WRITE_TRUNCATE",
    autodetect: bool = True,
    max_bad_records: int = 100
) -> bool:
    """
    Upload DataFrame to BigQuery with error handling.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to upload
    table_name : str
        BigQuery table name
    project : str
        GCP project ID
    dataset : str
        BigQuery dataset name
    client : bigquery.Client, optional
        BigQuery client (will create if not provided)
    write_disposition : str, optional
        Write mode (default: "WRITE_TRUNCATE")
    autodetect : bool, optional
        Auto-detect schema (default: True)
    max_bad_records : int, optional
        Maximum bad records to tolerate (default: 100)

    Returns:
    --------
    bool
        True if upload successful, False otherwise
    """
    table_id = f"{project}.{dataset}.{table_name}"

    print("\n" + "=" * 60)
    print(f"UPLOADING TO BIGQUERY: {table_name}")
    print("=" * 60)

    # Initialize client if not provided
    if client is None:
        client = get_bigquery_client(project)

    # Ensure dataset exists
    ensure_dataset_exists(client, project, dataset)

    # Display info
    print(f"\n📊 Data to upload:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Prepare DataFrame
    df_clean = prepare_dataframe_for_bigquery(df)

    # Convert to CSV in memory
    print("\n📄 Converting DataFrame to CSV...")
    csv_buffer = io.StringIO()
    df_clean.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_content = csv_buffer.getvalue().encode('utf-8')

    # Create job configuration
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=autodetect,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        max_bad_records=max_bad_records,
        encoding='UTF-8',
        allow_quoted_newlines=True
    )

    print(f"\n⬆️  Uploading {len(df_clean):,} rows to '{table_name}'...")

    # Upload from CSV
    try:
        file_obj = io.BytesIO(csv_content)
        job = client.load_table_from_file(file_obj, table_id, job_config=job_config)
        job.result()  # Wait for completion

        # Verify upload
        table = client.get_table(table_id)
        print(f"\n✅ SUCCESS!")
        print(f"   Table: {table_id}")
        print(f"   Rows uploaded: {table.num_rows:,}")

        # Safe access to optional attributes
        if hasattr(table, 'num_bytes') and table.num_bytes is not None:
            print(f"   Table size: {table.num_bytes / (1024*1024):.2f} MB")
        if hasattr(table, 'created') and table.created is not None:
            print(f"   Created: {table.created.strftime('%Y-%m-%d %H:%M:%S')}")

        # Show schema preview
        if table.schema:
            print(f"\n📐 Schema preview (first 5 columns):")
            for i, field in enumerate(table.schema[:5], 1):
                print(f"   {i}. {field.name:20} : {field.field_type}")
            if len(table.schema) > 5:
                print(f"   ... and {len(table.schema) - 5} more columns")

        return True

    except Exception as e:
        print(f"\n❌ CSV upload failed: {e}")

        # Try alternative method (direct DataFrame upload)
        print("\n🔄 Trying alternative upload method...")
        try:
            direct_job_config = bigquery.LoadJobConfig(
                write_disposition=write_disposition,
                autodetect=autodetect,
                max_bad_records=max_bad_records
            )

            direct_job = client.load_table_from_dataframe(df_clean, table_id, job_config=direct_job_config)
            direct_job.result()

            table = client.get_table(table_id)
            print(f"✅ Direct upload successful!")
            print(f"   Rows uploaded: {table.num_rows:,}")
            return True

        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            print(f"\n💡 Suggestion: Save data locally and upload manually via Google Cloud Console")
            return False


def extract_coordinates_from_dict(df: pd.DataFrame, geo_column: str = 'geo_point_2d') -> pd.DataFrame:
    """
    Extract lat/lon from dictionary-format geo_point_2d column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    geo_column : str, optional
        Name of the geo column (default: 'geo_point_2d')

    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted lat/lon columns
    """
    df_clean = df.copy()

    if geo_column not in df_clean.columns:
        return df_clean

    # Check format of first non-null value
    sample = df_clean[geo_column].dropna().iloc[0] if not df_clean[geo_column].dropna().empty else None

    if sample is None:
        return df_clean

    if isinstance(sample, dict):
        # Format: {'lon': x, 'lat': y}
        df_clean['lon'] = df_clean[geo_column].apply(
            lambda x: float(x['lon']) if isinstance(x, dict) and 'lon' in x else None
        )
        df_clean['lat'] = df_clean[geo_column].apply(
            lambda x: float(x['lat']) if isinstance(x, dict) and 'lat' in x else None
        )
    elif isinstance(sample, list):
        # Format: [lat, lon]
        df_clean['lat'] = df_clean[geo_column].apply(
            lambda x: float(x[0]) if isinstance(x, list) and len(x) > 0 else None
        )
        df_clean['lon'] = df_clean[geo_column].apply(
            lambda x: float(x[1]) if isinstance(x, list) and len(x) > 1 else None
        )

    valid_coords = df_clean['lon'].notna().sum()
    print(f"✅ Extracted coordinates for {valid_coords} rows")

    return df_clean


def clean_duplicates(df: pd.DataFrame, strategy: str = 'coordinates', lat_col: str = 'lat', lon_col: str = 'lon') -> pd.DataFrame:
    """
    Remove duplicates based on different strategies.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str, optional
        'coordinates', 'address', or 'strict' (default: 'coordinates')
    lat_col : str, optional
        Latitude column name (default: 'lat')
    lon_col : str, optional
        Longitude column name (default: 'lon')

    Returns:
    --------
    pd.DataFrame
        Deduplicated DataFrame
    """
    df_work = df.copy()
    original_count = len(df_work)

    # Ensure coordinate columns exist and are rounded for comparison
    if lat_col in df_work.columns and lon_col in df_work.columns:
        df_work[f'{lat_col}_round'] = df_work[lat_col].apply(lambda x: round(x, 6) if pd.notna(x) else None)
        df_work[f'{lon_col}_round'] = df_work[lon_col].apply(lambda x: round(x, 6) if pd.notna(x) else None)

    if strategy == 'coordinates':
        df_deduped = df_work.drop_duplicates(subset=[f'{lat_col}_round', f'{lon_col}_round'], keep='first')
    elif strategy == 'address':
        if 'adresse' in df_work.columns or 'address' in df_work.columns:
            addr_col = 'adresse' if 'adresse' in df_work.columns else 'address'
            commune_col = 'commune' if 'commune' in df_work.columns else None

            if commune_col:
                df_deduped = df_work.drop_duplicates(subset=[addr_col, commune_col], keep='first')
            else:
                df_deduped = df_work.drop_duplicates(subset=[addr_col], keep='first')
        else:
            df_deduped = df_work
    elif strategy == 'strict':
        df_deduped = df_work.drop_duplicates(keep='first')
    else:
        raise ValueError("Strategy must be 'coordinates', 'address', or 'strict'")

    # Clean up temporary columns
    df_deduped = df_deduped.drop(columns=[f'{lat_col}_round', f'{lon_col}_round'], errors='ignore')

    removed_count = original_count - len(df_deduped)
    print(f"Original rows: {original_count:,}")
    print(f"After {strategy} deduplication: {len(df_deduped):,}")
    print(f"Removed {removed_count:,} duplicates")

    return df_deduped
