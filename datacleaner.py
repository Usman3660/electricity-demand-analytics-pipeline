"""
Data Preprocessing Script

This script performs the following tasks on the merged data CSV file:

1) Data Cleaning and Consistency
   a) Missing Data
   b) Data Type Conversions
   c) Handling Duplicates and Inconsistencies
2) Feature Engineering
   a) Deriving new features from timestamps
   b) Normalizing numerical features
3) Documentation
   a) Inline comments describing assumptions and decisions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1. Load the merged data
# =============================================================================

# Replace this with your actual merged file name or path
merged_file = "./merged_data_20250302_011348.csv"  # e.g. merged_data_20250302_141530.csv
df = pd.read_csv(merged_file)

# Make a copy to avoid modifying the original DataFrame directly
data = df.copy()

# =============================================================================
# 2. Data Cleaning and Consistency
# =============================================================================

# -----------------------------------------------------------------------------
# a) Missing Data
# -----------------------------------------------------------------------------

# Calculate the number of missing values and their percentage for each column
missing_counts = data.isnull().sum()
missing_percent = (missing_counts / len(data)) * 100
missing_summary = pd.concat([missing_counts, missing_percent.rename("percentage")], axis=1)
print("=== Missing Data Summary per Column ===")
print(missing_summary)

# Decide if missingness is MCAR, MAR, or MNAR:
# - MCAR (Missing Completely at Random): No pattern to the missingness.
# - MAR (Missing At Random): Missingness can be explained by other observed variables.
# - MNAR (Missing Not At Random): The missingness is related to the value itself.
#
# For a generic dataset, we often lack domain-specific knowledge to definitively classify.
# Here, we'll assume missing values are either MCAR or MAR for simplicity.

# Example strategy:
# - For numerical columns: Impute missing values with the mean.
# - For categorical columns: Impute missing values with the mode.
# - If the missing percentage is extremely high, we might drop the column or use advanced imputation.

for col in data.columns:
    if data[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(data[col]):
            mean_val = data[col].mean()
            data[col].fillna(mean_val, inplace=True)
            print(f"[Imputation] Column '{col}': filled NaN with mean ({mean_val:.2f}).")
        else:
            mode_val = data[col].mode(dropna=True)
            if len(mode_val) > 0:
                mode_val = mode_val[0]
                data[col].fillna(mode_val, inplace=True)
                print(f"[Imputation] Column '{col}': filled NaN with mode ('{mode_val}').")
            else:
                # If the column is entirely NaN, mode() is empty. We might drop or keep as is.
                data[col].fillna("Unknown", inplace=True)
                print(f"[Imputation] Column '{col}': all values missing; replaced with 'Unknown'.")

# -----------------------------------------------------------------------------
# b) Data Type Conversions
# -----------------------------------------------------------------------------
# Identify columns that should be numeric or datetime, and convert them accordingly.

# Example: If your flattened JSON has a "period" or "timestamp" column
# that contains date/time strings, convert to datetime.
datetime_columns = ["period", "timestamp"]  # adjust these names if needed

for col in datetime_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        print(f"[Type Conversion] Column '{col}' converted to datetime.")

# Convert any numeric columns stored as objects to numeric
for col in data.select_dtypes(include=['object']).columns:
    # Attempt numeric conversion
    try:
        data[col] = pd.to_numeric(data[col])
        print(f"[Type Conversion] Column '{col}' converted to numeric.")
    except ValueError:
        # If conversion fails, we assume the column is truly categorical/string
        pass

# -----------------------------------------------------------------------------
# c) Handling Duplicates and Inconsistencies
# -----------------------------------------------------------------------------
# Remove duplicate rows
initial_shape = data.shape
data.drop_duplicates(inplace=True)
duplicates_removed = initial_shape[0] - data.shape[0]
print(f"[Duplicates] Removed {duplicates_removed} duplicate rows.")

# Identify outliers (for numerical columns) using the IQR method and remove them.
numeric_cols = data.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    # Use Interquartile Range (IQR) to detect outliers
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    count_before = data.shape[0]
    # Keep only rows within the IQR bounds
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    count_after = data.shape[0]
    outliers_removed = count_before - count_after
    if outliers_removed > 0:
        print(f"[Outliers] Column '{col}': removed {outliers_removed} outlier rows.")

# =============================================================================
# 3. Feature Engineering
# =============================================================================
# For any datetime column, create additional time-based features.
# Example: If you have a 'timestamp' column or 'period' column.

time_col = None
for candidate in ["period", "timestamp"]:
    if candidate in data.columns and pd.api.types.is_datetime64_any_dtype(data[candidate]):
        time_col = candidate
        break

if time_col:
    data["hour"] = data[time_col].dt.hour
    data["day"] = data[time_col].dt.day
    data["month"] = data[time_col].dt.month
    data["day_of_week"] = data[time_col].dt.dayofweek  # Monday=0, Sunday=6
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Simple season determination (northern hemisphere):
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    data["season"] = data["month"].apply(get_season)
    print("[Feature Engineering] Extracted hour, day, month, day_of_week, is_weekend, and season.")

# Example: If you have a holiday list or logic, you can create a holiday flag here (not implemented).

# -----------------------------------------------------------------------------
# Normalize/Standardize numerical features if needed
# -----------------------------------------------------------------------------
# For demonstration, we'll apply Min-Max scaling to numeric columns except for time-based columns.
exclude_cols = ["hour", "day", "month", "day_of_week", "is_weekend"]
scale_cols = [c for c in numeric_cols if c not in exclude_cols]

if scale_cols:
    scaler = MinMaxScaler()
    data[scale_cols] = scaler.fit_transform(data[scale_cols])
    print(f"[Feature Engineering] Normalized columns: {scale_cols}")

# =============================================================================
# 4. Documentation
# =============================================================================
# Inline comments and print statements have explained:
# - Our assumptions for missing data handling (MCAR/MAR assumption).
# - Type conversions for datetime and numeric columns.
# - Duplicate removal and outlier detection (IQR method).
# - Additional features derived from datetime columns.
# - Normalization of numeric columns.

# =============================================================================
# 5. Save the Processed Data
# =============================================================================

# Save to a new CSV file (e.g., "processed_data.csv").
processed_file = "processed_data.csv"
data.to_csv(processed_file, index=False)
print(f"[Save] Processed data saved to '{processed_file}'.")

# End of script
