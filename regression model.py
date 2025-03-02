import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# 1. Load Processed Data
# ------------------------------------------------------------------------------
processed_file = "processed_data.csv"
data = pd.read_csv(processed_file)

# ------------------------------------------------------------------------------
# 2. Feature Selection
# ------------------------------------------------------------------------------
# We assume the processed data has time-based features (e.g., 'hour', 'day', 'month', 'day_of_week').
# Additionally, if available, we include 'temperature_of_day'.
selected_features = []
for col in ["hour", "day", "month", "day_of_week"]:
    if col in data.columns:
        selected_features.append(col)
if "temperature_of_day" in data.columns:
    selected_features.append("temperature_of_day")

print("Selected features for modeling:", selected_features)

# Define the target variable; here we assume 'value' represents electricity demand.
target = "value"
if target not in data.columns:
    raise ValueError("Target column 'value' not found in the dataset.")

# Create feature matrix X and target vector y
X = data[selected_features]
y = data[target]

# ------------------------------------------------------------------------------
# 3. Model Development
# ------------------------------------------------------------------------------
# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# ------------------------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------------------------
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# Plot Actual vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2, label="Ideal Fit")
plt.xlabel("Actual Electricity Demand")
plt.ylabel("Predicted Electricity Demand")
plt.title("Actual vs. Predicted Electricity Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 5. Residual Analysis
# ------------------------------------------------------------------------------
# Compute residuals
residuals = y_test - y_pred

# Plot histogram and density of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot residuals vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color="green")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Electricity Demand")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Documentation and Rationale
# ------------------------------------------------------------------------------
"""
Feature Selection:
- We selected time-based features (hour, day, month, day_of_week) as they are known to affect electricity demand.
- Optionally, 'temperature_of_day' is included if available since ambient temperature can influence consumption.

Model Development:
- The dataset is split into 80% training and 20% testing sets.
- A simple Linear Regression model is employed due to its interpretability and ease of use.

Evaluation:
- MSE, RMSE, and R2 score provide a measure of prediction accuracy.
- The Actual vs. Predicted plot visually assesses model performance.
- Residual analysis (via histogram and scatter plot) is performed to ensure residuals are randomly distributed around zero, indicating a good fit.

Technical Rationale:
- Linear Regression is an effective starting point for regression tasks. However, if non-linear relationships are detected, more complex models (e.g., Random Forest or Gradient Boosting) could be explored.
- Residual plots help diagnose issues such as heteroscedasticity or non-linearity, guiding further model improvements if needed.
"""

