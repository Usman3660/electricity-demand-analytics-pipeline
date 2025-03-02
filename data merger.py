import os
import glob
import json
import pandas as pd
import logging
from datetime import datetime

# ------------------------------------------------------------------------------
# Logging Setup: Log both to the console and a file with a timestamped name.
# ------------------------------------------------------------------------------
log_filename = f"data_merging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logging.info("Starting data merging process...")

# ------------------------------------------------------------------------------
# Define the base folder containing raw data files (CSV/JSON) in nested directories.
# ------------------------------------------------------------------------------
data_folder = 'raw'  # Change if your raw data folder is named differently

# Initialize a list to hold DataFrames from each file.
df_list = []

# ------------------------------------------------------------------------------
# Utility Function: Standardize column names (trim spaces, lower case, replace spaces with underscores).
# ------------------------------------------------------------------------------
def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

# ------------------------------------------------------------------------------
# Recursively Load CSV Files
# ------------------------------------------------------------------------------
csv_files = glob.glob(os.path.join(data_folder, '**', '*.csv'), recursive=True)
if not csv_files:
    logging.warning("No CSV files found in any subfolder of the specified folder.")
else:
    for file in csv_files:
        logging.info(f"Loading CSV file: {file}")
        try:
            # Attempt to load CSV with UTF-8 encoding.
            df_csv = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try with latin1 encoding.
            logging.warning(f"Encoding issue with {file}. Trying 'latin1' encoding.")
            df_csv = pd.read_csv(file, encoding='latin1')

        # Standardize column names.
        df_csv = standardize_columns(df_csv)
        logging.info(f"Loaded {file} with {df_csv.shape[0]} records and {df_csv.shape[1]} columns.")
        df_list.append(df_csv)

# ------------------------------------------------------------------------------
# Recursively Load JSON Files
# ------------------------------------------------------------------------------
json_files = glob.glob(os.path.join(data_folder, '**', '*.json'), recursive=True)
if not json_files:
    logging.warning("No JSON files found in any subfolder of the specified folder.")
else:
    for file in json_files:
        logging.info(f"Loading JSON file: {file}")
        try:
            # Load the raw JSON manually so we can flatten the "response.data" structure.
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Flatten the JSON using pd.json_normalize
            # record_path points to the list we want to explode into rows (response -> data)
            # meta extracts higher-level keys from response
            df_json = pd.json_normalize(
                data,
                record_path=["response", "data"],
                meta=[
                    ["response", "total"],
                    ["response", "dateFormat"],
                    ["response", "frequency"]
                ],
                errors="ignore"
            )
            
            # Clean up column names to remove dots, etc.
            df_json.columns = [col.replace(".", "_") for col in df_json.columns]
            
            # Convert "value" to numeric if it's present
            if "value" in df_json.columns:
                df_json["value"] = pd.to_numeric(df_json["value"], errors="coerce")

            # Optionally standardize columns further (e.g., unify naming)
            df_json = standardize_columns(df_json)

            logging.info(f"Flattened JSON from {file} with {df_json.shape[0]} records and {df_json.shape[1]} columns.")
            df_list.append(df_json)

        except Exception as e:
            logging.error(f"Failed to load or flatten JSON from {file}: {e}")

# ------------------------------------------------------------------------------
# Merge/Concatenate all DataFrames
# ------------------------------------------------------------------------------
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Merged DataFrame has {merged_df.shape[0]} records and {merged_df.shape[1]} columns.")
else:
    logging.error("No data files found or successfully loaded. The merged DataFrame will be empty.")
    merged_df = pd.DataFrame()

# ------------------------------------------------------------------------------
# Save the Merged Data with a Versioned File Name
# ------------------------------------------------------------------------------
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_file = f"merged_data_{timestamp_str}.csv"
merged_df.to_csv(merged_file, index=False)
logging.info(f"Merged data saved to {merged_file}")
logging.info("Data merging process completed.")
