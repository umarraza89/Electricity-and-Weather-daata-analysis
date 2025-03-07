import os
import pandas as pd

# 📂 Paths
BASE_FOLDER = "C:/Users/umar/Desktop/datasciencee/raw"
MERGED_FILE = os.path.join(BASE_FOLDER, "merged_dataset.csv")  # Read from merged file
PROCESSED_FILE = os.path.join(BASE_FOLDER, "newmerged.csv")  # New processed dataset

def analyze_data(df):
    """Analyze data before cleaning"""
    print("\n🔹 Initial Data Analysis...")
    
    # Print missing data percentage
    missing_percent = df.isnull().sum() / len(df) * 100
    print("\n❓ Missing Data Percentage (Before Cleaning):")
    print(missing_percent)
    
    # Print data types
    print("\n🗂️ Data Types (Before Cleaning):")
    print(df.dtypes)
    
    # Check duplicates (since we already removed them earlier, this will likely be 0)
    duplicate_count = df.duplicated().sum()
    print(f"\n⚠️ Duplicate Rows Found (If any left): {duplicate_count}")

def clean_data(df):
    """Handles missing values, data types, outliers, and feature engineering."""
    print("\n🔹 Cleaning Data...")

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")  # Fill missing categorical values
        else:
            df[col] = df[col].fillna(df[col].median())  # Fill numerical with median

    # Convert categorical columns
    df['parent'] = df['parent'].astype('category')
    df['subba'] = df['subba'].astype('category')

    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Feature Engineering (Extract Time Features)
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Outlier Detection & Removal using IQR
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers_before = df.shape[0]
    
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    outliers_removed = outliers_before - df.shape[0]

    print(f"\n✅ Cleaning Complete! Removed {outliers_removed} Outliers.")

    return df

def post_clean_analysis(df):
    """Analyze data after cleaning"""
    print("\n📊 Data Summary After Cleaning...")

    # Print missing data percentage after cleaning
    missing_percent = df.isnull().sum() / len(df) * 100
    print("\n❓ Missing Data Percentage (After Cleaning):")
    print(missing_percent)

    # Print updated data types
    print("\n🗂️ Data Types (After Cleaning):")
    print(df.dtypes)

    print(f"\n📂 Final Processed Dataset Shape: {df.shape}")

def main():
    # Load merged data
    if not os.path.exists(MERGED_FILE):
        print("❌ Merged dataset not found!")
        return
    
    merged_df = pd.read_csv(MERGED_FILE)
    
    # Analyze before cleaning
    analyze_data(merged_df)

    # Clean & process data
    processed_df = clean_data(merged_df)
    
    # Analyze after cleaning
    post_clean_analysis(processed_df)
    
    # Save cleaned data
    processed_df.to_csv(PROCESSED_FILE, index=False)
    print(f"\n✅ Processed data saved: {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
