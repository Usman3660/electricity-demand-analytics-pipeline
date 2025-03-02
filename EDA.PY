import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# =============================================================================
# 1. Load Processed Data
# =============================================================================

processed_file = "./processed_data.csv"
if not os.path.exists(processed_file):
    raise FileNotFoundError(f"Processed data file '{processed_file}' not found. Please check the file path.")

# Load the processed data
data = pd.read_csv(processed_file)

# Check for a datetime column (adjust if your date column is named differently)
date_col = None
for col in ["timestamp", "period"]:
    if col in data.columns:
        date_col = col
        break

if date_col:
    # Convert to datetime if not already done and sort the data
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data.sort_values(by=date_col, inplace=True)
    print(f"Using '{date_col}' as the time index.")
else:
    print("Warning: No datetime column found. Time series analysis will be skipped.")

# Assuming the electricity demand is stored in the column "value".
# Adjust the column name if needed.
demand_col = "value"
if demand_col not in data.columns:
    raise ValueError(f"Column '{demand_col}' not found in the data. Please ensure the electricity demand column is named correctly.")

# =============================================================================
# 2. Exploratory Data Analysis (EDA)
# =============================================================================

# -----------------------
# Statistical Summary
# -----------------------

print("\n=== Statistical Summary for Numerical Features ===")
# Compute basic descriptive statistics
desc_stats = data.describe().T
# Calculate additional statistics
desc_stats['median'] = data.median(numeric_only=True)
desc_stats['skewness'] = data.skew(numeric_only=True)
desc_stats['kurtosis'] = data.kurtosis(numeric_only=True)
print(desc_stats)

# -----------------------
# Time Series Analysis
# -----------------------

if date_col:
    plt.figure(figsize=(12, 6))
    plt.plot(data[date_col], data[demand_col], label="Electricity Demand", color="blue", linewidth=1)
    plt.title("Electricity Demand Over Time")
    plt.xlabel("Time")
    plt.ylabel("Demand")
    plt.grid(True)
    # Optional: annotate key trends or irregularities. Example: Mark a specific date
    # plt.axvline(pd.to_datetime("2022-01-01"), color='red', linestyle='--', label="Example Annotation")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Skipping time series plot due to missing datetime column.")

# -----------------------
# Univariate Analysis
# -----------------------

# Histogram for electricity demand
plt.figure(figsize=(8, 5))
sns.histplot(data[demand_col], kde=False, bins=30, color="skyblue")
plt.title("Histogram of Electricity Demand")
plt.xlabel("Demand")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot for electricity demand
plt.figure(figsize=(8, 5))
sns.boxplot(x=data[demand_col], color="lightgreen")
plt.title("Boxplot of Electricity Demand")
plt.xlabel("Demand")
plt.tight_layout()
plt.show()

# Density Plot for electricity demand
plt.figure(figsize=(8, 5))
sns.kdeplot(data[demand_col], shade=True, color="orange")
plt.title("Density Plot of Electricity Demand")
plt.xlabel("Demand")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# -----------------------
# Correlation Analysis
# -----------------------

# Compute correlation matrix for numerical features
corr_matrix = data.select_dtypes(include=[np.number]).corr()
print("\n=== Correlation Matrix ===")
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Discuss potential multicollinearity: 
# (Typically, correlations above 0.8 in absolute value may indicate multicollinearity.)

# -----------------------
# Advanced Time Series Techniques
# -----------------------

if date_col:
    # Set the datetime column as index for time series decomposition
    ts_data = data.set_index(date_col)
    # Resample if necessary (e.g., to daily averages if data is hourly)
    # For this example, we assume the data is at a frequency suitable for decomposition.
    try:
        decomposition = seasonal_decompose(ts_data[demand_col], model='additive', period=24)
        fig = decomposition.plot()
        fig.set_size_inches(12, 10)
        plt.suptitle("Seasonal Decomposition of Electricity Demand", fontsize=16)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Time series decomposition failed: {e}")

    # Augmented Dickey-Fuller (ADF) Test for stationarity
    adf_result = adfuller(ts_data[demand_col].dropna())
    print("\n=== Augmented Dickey-Fuller Test ===")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    for key, value in adf_result[4].items():
        print(f"Critical Value {key}: {value:.4f}")
    if adf_result[1] < 0.05:
        print("The time series is likely stationary (reject null hypothesis).")
    else:
        print("The time series is likely non-stationary (fail to reject null hypothesis).")
else:
    print("Skipping advanced time series analysis due to missing datetime column.")
