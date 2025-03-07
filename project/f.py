import os
import pandas as pd
import json


BASE_FOLDER = "C:/Users/umar/Desktop/datasciencee/raw"
ELECTRICITY_FOLDER = os.path.join(BASE_FOLDER, "electricity_raw_data")
WEATHER_FOLDER = os.path.join(BASE_FOLDER, "weather_raw_data")
OUTPUT_FILE = os.path.join(BASE_FOLDER, "merged_dataset.csv")

def load_electricity_data(folder):
    """Loads electricity data from all JSON files in the given folder."""
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')]
    
    if not all_files:
        print("❌ No electricity data files found!")
        return pd.DataFrame()

    data_list = []
    for file in all_files:
        with open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for entry in json_data.get("response", {}).get("data", []):
                data_list.append({
                    "datetime": entry.get("period"),
                    "subba": entry.get("subba"),
                    "parent": entry.get("parent"),
                    "value": entry.get("value")
                })

    electricity_df = pd.DataFrame(data_list)

    electricity_df['datetime'] = pd.to_datetime(electricity_df['datetime'], errors='coerce')

    return electricity_df

def load_weather_data(folder):
    """Loads weather data from all CSV files in the given folder."""
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

    if not all_files:
        print("❌ No weather data files found!")
        return pd.DataFrame()

    df_list = [pd.read_csv(f) for f in all_files]
    weather_df = pd.concat(df_list, ignore_index=True)

    weather_df.rename(columns={"date": "datetime"}, inplace=True)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], errors='coerce', utc=True)

    # 🚀 Handle missing temperature values (Fill NaN with previous value)
    weather_df['temperature_2m'] = weather_df['temperature_2m'].ffill()

    return weather_df

def log_dataset_info(df, name):
    """Logs dataset statistics before merging, keeping duplicates for review."""
    print(f"\n📊 {name} Data Summary:")
    print(f"📝 Total Records: {len(df):,}")
    print(f"🔢 Features: {list(df.columns)}")

    
    missing_values = df.replace(["", " ", "NaN", "None"], pd.NA).isnull().sum()
    print(f"❓ Missing values:\n{missing_values}")

    # 🔄 Check duplicates (both full and partial)
    full_duplicates = df.duplicated().sum()
    partial_duplicates = df.duplicated(subset=['datetime']).sum()

    print(f"🗑️ Full Duplicates: {full_duplicates}")
    print(f"🔄 Partial Duplicates (datetime only): {partial_duplicates}")

def main():
 
    electricity_df = load_electricity_data(ELECTRICITY_FOLDER)
    weather_df = load_weather_data(WEATHER_FOLDER)

    log_dataset_info(electricity_df, "ELECTRICITY")
    log_dataset_info(weather_df, "WEATHER")

    # ✅ Standardize datetime formats
    electricity_df['datetime'] = pd.to_datetime(electricity_df['datetime'], errors='coerce').dt.tz_localize('UTC')

    merged_df = pd.merge(
        electricity_df,
        weather_df,
        on='datetime',
        how='inner'
    ).sort_values('datetime')

   
    log_dataset_info(merged_df, "MERGED (Before Removing Duplicates)")

   
    merged_df = merged_df.drop_duplicates()

    # ✅ Log final dataset summary
    log_dataset_info(merged_df, "MERGED (Final)")

    print("\n✅ Merge Successful!")
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
