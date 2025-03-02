import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time  # Import time module for delays

# =============================================================================
# 1. Load Processed Data
# =============================================================================

processed_file = "processed_data.csv"
if not os.path.exists(processed_file):
    raise FileNotFoundError(f"File '{processed_file}' not found. Please check the path.")

# Load the processed data
data = pd.read_csv(processed_file)

# For demonstration, assume electricity demand is in the "value" column.
if "value" not in data.columns:
    raise ValueError("The column 'value' (electricity demand) is not in the dataset.")

# Make a copy to preserve original data for comparison
df_original = data.copy()

# =============================================================================
# 2. Outlier Detection
# =============================================================================

# --- IQR-based Detection ---
def iqr_outlier_bounds(series, factor=1.5):
    """Calculate lower and upper bounds using the IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return lower_bound, upper_bound

# Calculate IQR bounds for the "value" column
lower_iqr, upper_iqr = iqr_outlier_bounds(data["value"])
print(f"IQR Method: Lower Bound = {lower_iqr:.2f}, Upper Bound = {upper_iqr:.2f}")

# Flag outliers using IQR method
data["outlier_iqr"] = ((data["value"] < lower_iqr) | (data["value"] > upper_iqr))

# --- Z-score Method ---
def zscore_outliers(series, threshold=3):
    """Return a boolean mask where True indicates an outlier using Z-score."""
    z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
    return z_scores > threshold

# Compute Z-score mask for the "value" column
data["outlier_zscore"] = zscore_outliers(data["value"])
n_outliers_z = data["outlier_zscore"].sum()
print(f"Z-score Method: Found {n_outliers_z} outliers using threshold |Z| > 3.")

# =============================================================================
# 3. Handling Outliers
# =============================================================================

# Evaluate impact before handling
print("\nBefore handling outliers:")
print(data["value"].describe())

# Technical Rationale:
# In many real-world scenarios (like electricity demand), extreme values may be valid but can skew analysis.
# Instead of removing outliers (which might discard important information), we choose winsorization (capping).
# Here we cap values at the IQR-determined lower and upper bounds.

# Create a new column for winsorized values
data["value_winsor"] = data["value"].copy()

# Cap values below lower bound to lower_iqr and above upper bound to upper_iqr
data.loc[data["value_winsor"] < lower_iqr, "value_winsor"] = lower_iqr
data.loc[data["value_winsor"] > upper_iqr, "value_winsor"] = upper_iqr

print("\nAfter winsorization (capping using IQR bounds):")
print(data["value_winsor"].describe())

# =============================================================================
# 4. Visualizations: Before and After Outlier Handling
# =============================================================================

# Function to plot histograms and boxplots for a given column
def plot_distribution(df, column, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[column], bins=30, ax=axes[0], color="skyblue", kde=True)
    axes[0].set_title(f"Histogram of {column} {title_suffix}")
    axes[0].set_xlabel(column)
    
    sns.boxplot(x=df[column], ax=axes[1], color="lightgreen")
    axes[1].set_title(f"Boxplot of {column} {title_suffix}")
    axes[1].set_xlabel(column)
    
    plt.tight_layout()
    plt.show()

# Plot original "value" distribution
plot_distribution(df_original, "value", title_suffix="(Original Data)")

# Wait for 5 seconds before displaying the next plot
time.sleep(5)

# Plot winsorized "value" distribution
plot_distribution(data, "value_winsor", title_suffix="(Winsorized Data)")

# =============================================================================
# 5. Documentation Summary
# =============================================================================
"""
Outlier Detection and Handling Summary:
- IQR Method: We computed the interquartile range for the 'value' column and flagged
  data points falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
- Z-score Method: We computed Z-scores and flagged observations with |Z| > 3.
  
Handling Strategy:
- We evaluated that removing outliers might lose important extreme values in electricity demand.
- Instead, we applied winsorization (capping) using the IQR bounds to limit the impact of extreme values while preserving overall data structure.
- Before-and-after visualizations (histograms and boxplots) are provided to show the effect of the outlier treatment.
  
This approach maintains the integrity of the dataset and reduces skewness due to extreme values, thereby improving the reliability of subsequent analysis.
"""
