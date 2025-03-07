import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# 📂 Define Paths
BASE_FOLDER = "C:/Users/umar/Desktop/datasciencee/raw"
MERGED_FILE = os.path.join(BASE_FOLDER, "newmerged.csv")
FINAL_FILE = os.path.join(BASE_FOLDER, "final_file.csv")  # New final output file

def load_data():
    """Loads the dataset and ensures it's valid."""
    if not os.path.exists(MERGED_FILE):
        raise FileNotFoundError(f"❌ File not found: {MERGED_FILE}")
    
    # Load the raw data first
    raw_df = pd.read_csv(MERGED_FILE, parse_dates=["datetime"])
    print(f"📊 Total Records Before Processing: {raw_df.shape[0]}")  # Show total records before preprocessing
    
    # Copy raw_df to avoid modifying original data
    df = raw_df.copy()
    
    # Preprocessing steps
    df = df.drop_duplicates(subset=["datetime"])  # Remove duplicate timestamps
    df = df.sort_values("datetime")  # Ensure chronological order
    
    print(f"✅ Data Loaded Successfully! Shape After Processing: {df.shape}")  # Show records after preprocessing
    
    return df

def detect_outliers_iqr(df, column):
    """Detects outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"🔍 IQR Outliers Detected in {column}: {len(outliers)}")
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column):
    """Detects outliers using the Z-score method."""
    z_scores = np.abs(zscore(df[column]))
    threshold = 3  # Common threshold for Z-score
    outliers = df[z_scores > threshold]
    
    print(f"🔍 Z-Score Outliers Detected in {column}: {len(outliers)}")
    return outliers, threshold

def plot_outliers(df, column, title=""):
    """Plots original vs. cleaned data distribution."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[column], color='red')
    plt.title(f'Before Outlier Removal: {column}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[column], bins=30, kde=True, color='blue')
    plt.title(f'Distribution: {column} {title}')

    plt.tight_layout()
    plt.show()

def handle_outliers(df, column):
    """Removes extreme outliers using both methods and evaluates impact."""
    print(f"\n⚙️ Handling Outliers for {column}")

    # Detect outliers
    iqr_outliers, lower_bound, upper_bound = detect_outliers_iqr(df, column)
    zscore_outliers, threshold = detect_outliers_zscore(df, column)

    # Show before cleaning
    plot_outliers(df, column, title="(Before Cleaning)")

    # Handling Strategy: Remove extreme outliers
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"✅ Outliers removed using IQR method: {len(df) - len(df_cleaned)}")

    # Show after cleaning
    plot_outliers(df_cleaned, column, title="(After Cleaning)")

    return df_cleaned

def main():
    """Runs outlier detection and handling."""
    df = load_data()

    numerical_columns = df.select_dtypes(include=['number']).columns

    for column in numerical_columns:
        df = handle_outliers(df, column)

    # Save cleaned data to final file
    df.to_csv(FINAL_FILE, index=False)
    print(f"\n📂 Final cleaned dataset saved as: {FINAL_FILE} ✅")
    print(f"✅ Final Dataset Shape: {df.shape}")

if __name__ == "__main__":
    main()
