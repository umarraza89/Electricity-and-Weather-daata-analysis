import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📂 Define Paths
BASE_FOLDER = "C:/Users/umar/Desktop/datasciencee/raw"
MERGED_FILE = os.path.join(BASE_FOLDER, "newmerged.csv")

def load_data():
    """Loads the dataset and ensures it's valid."""
    if not os.path.exists(MERGED_FILE):
        raise FileNotFoundError(f"❌ File not found: {MERGED_FILE}")
    
    df = pd.read_csv(MERGED_FILE, parse_dates=["datetime"])
    
    print(f"📊 Total Records Before Processing: {df.shape[0]}")  # Show total records before preprocessing
    
    df = df.drop_duplicates(subset=["datetime"])  # Remove duplicate timestamps
    df = df.sort_values("datetime")  # Ensure chronological order
    
    print(f"✅ Data Loaded Successfully! Shape After Processing: {df.shape}")  # Show records after preprocessing
    print("Columns in DataFrame:", df.columns.tolist())  # Debugging: Show available columns
    
    return df

def feature_engineering(df):
    """Extracts useful time-based features for regression."""
    
    # Extract time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.weekday

    # Ensure 'value' column exists
    if "value" not in df.columns:
        raise KeyError("❌ ERROR: 'value' column (electricity demand) is missing from the dataset!")

    # Drop NaNs from target column ('value')
    df = df.dropna(subset=["value"])

    # Define predictors and target
    predictors = ["hour", "day", "month", "day_of_week", "temperature_2m"]
    target = "value"
    
    return df, predictors, target

def train_regression_model(df, predictors, target):
    """Splits data and trains a regression model."""
    X = df[predictors]
    y = df[target]
    
    # Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """Calculates regression metrics and prints results."""
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 **Model Evaluation Metrics:**")
    print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
    print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"📈 R² Score: {r2:.4f}")

def plot_results(y_test, y_pred):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    plt.xlabel("Actual Demand (value)")
    plt.ylabel("Predicted Demand")
    plt.title("Actual vs. Predicted Electricity Demand")
    plt.show()

def residual_analysis(y_test, y_pred):
    """Plots residual distribution to check errors."""
    residuals = y_test - y_pred
    
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.xlabel("Residual Error")
    plt.title("Residual Error Distribution")
    plt.show()

    print("\n🔍 Residual Analysis Summary:")
    print(f"Mean of Residuals: {residuals.mean():.4f}")
    print(f"Standard Deviation of Residuals: {residuals.std():.4f}")

def main():
    """Runs the regression workflow."""
    df = load_data()
    df, predictors, target = feature_engineering(df)
    
    model, X_test, y_test, y_pred = train_regression_model(df, predictors, target)
    
    evaluate_model(y_test, y_pred)
    plot_results(y_test, y_pred)
    residual_analysis(y_test, y_pred)

if __name__ == "__main__":
    main()
