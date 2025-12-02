import pandas as pd

# Load the raw CSV data
df = pd.read_csv('lahore_complete_data_july_to_dec_2024.csv')

# Clean datetime column (handle format and errors)
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M', errors='coerce')

# Drop rows with invalid datetime or missing main_aqi
df = df.dropna(subset=['datetime', 'main_aqi'])

# Set datetime as index and resample to hourly (forward-fill missing values)
df.set_index('datetime', inplace=True)
df = df.resample('H').mean().ffill()

# Save the cleaned data
df.to_csv('cleaned_lahore_aqi.csv')
print("Data cleaned and saved successfully. Rows in cleaned data:", len(df))