import pandas as pd
import numpy as np


#reading and loading the dataset
df_flights = pd.read_csv('flight_delays_data (1) - flight_delays_data (1).csv')
print(df_flights.head())

#basic cleaning and exploration
print(df_flights.info())
print(df_flights.isnull().sum())
#-----since 1714 rows have null values in 'Airline', we will fill them with 'Unknown' as the dataset of missing value is very low-----
df_flights['Airline'].fillna('Unknown', inplace=True)


print(df_flights.describe())
df_flights['flight_date'] = pd.to_datetime(df_flights['flight_date'], errors='coerce')
df_flights['Week'] = pd.to_numeric(df_flights['Week'], errors='coerce')

#extracting month and day from flight_date
df_flights['Year'] = df_flights['flight_date'].dt.year
df_flights['Month'] = df_flights['flight_date'].dt.month
df_flights['Dayofweek'] = df_flights['flight_date'].dt.dayofweek

#saving cleaned data to a parquet file
df_flights.to_parquet('cleaned_flight_delays.parquet', index=False)

print("Data cleaning completed and saved to 'cleaned_flight_delays.parquet'")

