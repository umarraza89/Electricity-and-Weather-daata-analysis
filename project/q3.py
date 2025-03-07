import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Define paths
BASE_FOLDER = "C:/Users/umar/Desktop/datasciencee/raw"
MERGED_FILE = os.path.join(BASE_FOLDER, "newmerged.csv")  # Latest cleaned data

def load_data():
    """Loads the latest merged CSV file."""
    if not os.path.exists(MERGED_FILE):
        raise FileNotFoundError(f"❌ File not found: {MERGED_FILE}")
    
    df = pd.read_csv(MERGED_FILE, parse_dates=["datetime"])
    df = df.drop_duplicates(subset=["datetime"])  # Remove duplicate timestamps
    df = df.sort_values("datetime")  # Ensure chronological order
    
    print(f"✅ Data Loaded Successfully from {MERGED_FILE}!")
    print(f"🔹 Dataset Shape: {df.shape}")  # Verify correct row count
    return df

def statistical_summary(df):
    """Computes key statistical metrics for numerical columns."""
    numerical_cols = df.select_dtypes(include=['number'])
    stats = numerical_cols.describe().T
    stats["skewness"] = numerical_cols.skew()
    stats["kurtosis"] = numerical_cols.kurtosis()
    
    print("\n🔹 Statistical Summary of Numerical Features:")
    print(stats)

def plot_time_series(df):
    """Plots the electricity demand over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["datetime"], df["value"], label="Electricity Demand", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Electricity Demand")
    plt.title("Electricity Demand Over Time")
    plt.legend()
    plt.grid()
    plt.show(block=True)  # Ensures display

def univariate_analysis(df):
    """Generates histograms, boxplots, and density plots for numerical features."""
    numerical_cols = df.select_dtypes(include=['number'])
    
    for col in numerical_cols:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        sns.histplot(df[col], bins=30, kde=True, color='blue')
        plt.title(f'Histogram of {col}')

        plt.subplot(1, 3, 2)
        sns.boxplot(y=df[col], color='green')
        plt.title(f'Boxplot of {col}')

        plt.subplot(1, 3, 3)
        sns.kdeplot(df[col], fill=True, color='red')
        plt.title(f'Density Plot of {col}')

        plt.tight_layout()
        plt.show(block=True)

def correlation_analysis(df):
    """Computes and visualizes the correlation matrix."""
    numerical_cols = df.select_dtypes(include=['number'])
    corr_matrix = numerical_cols.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show(block=True)

def time_series_decomposition(df):
    """Performs time series decomposition to analyze trend and seasonality."""
    df = df.set_index("datetime")
    df = df.asfreq("h")  # Ensures uniform time intervals with lowercase 'h'
    
    df["value"] = df["value"].interpolate()  # Fills missing values
    print(f"🔹 Missing Values After Interpolation: {df['value'].isna().sum()}")  # Debugging

    decomposition = sm.tsa.seasonal_decompose(df["value"], model="additive", period=24)

    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(df["value"], label="Original", color="blue")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label="Trend", color="green")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label="Seasonality", color="red")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label="Residuals", color="purple")
    plt.legend()

    plt.tight_layout()
    plt.show(block=True)

def test_stationarity(df):
    """Performs Augmented Dickey-Fuller test to check for stationarity."""
    result = adfuller(df["value"].dropna())

    print("\n📌 Augmented Dickey-Fuller Test Results:")
    print(f"Test Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")

    if result[1] < 0.05:
        print("✅ The data is stationary (reject null hypothesis).")
    else:
        print("❌ The data is non-stationary (fail to reject null hypothesis).")

def main():
    """Runs the full EDA process."""
    
    df = load_data()
    statistical_summary(df)
    plot_time_series(df)
    univariate_analysis(df)
    correlation_analysis(df)
    time_series_decomposition(df)
    test_stationarity(df)

if __name__ == "__main__":
    main()
