import pandas as pd
import numpy as np


# Load cleaned data
df_cleaned_flights = pd.read_parquet('cleaned_flight_delays.parquet')
print(df_cleaned_flights.head())

# Exploratory Data Analysis (EDA)
#here is_claim has two values 0 and 800,which might create a lot of issue in statistics or model building,so we will convert 800 to 1 for better understanding
df_cleaned_flights['is_claim_flag'] = df_cleaned_flights[['is_claim']].applymap(lambda x: 1 if x == 800 else 0)
overall_claim_rate = df_cleaned_flights['is_claim_flag'].mean()
print(f"Overall Claim Rate: {overall_claim_rate:.2%}")


#claim rates by airline
airline_claim_rates = (df_cleaned_flights.groupby('Airline')['is_claim_flag'].mean().reset_index().rename(columns={'is_claim_flag': 'airline_claim_rate'}))
print("Airline Claim Rates:")
print(airline_claim_rates.head(15))


#bucketizing claim rates to find out the price strategy by using qcut to create 3 buckets: low, medium, high
airline_claim_rates['airline_risk_bucket'] = pd.qcut(
    airline_claim_rates['airline_claim_rate'],
    q=[0, 0.2, 0.8, 1.0],
    labels=['Low', 'Medium', 'High']
)

#merge bucket info back to main dataframe to create row wise airline risk bucket
df_with_airline_bucket = df_cleaned_flights.merge(
    airline_claim_rates[['Airline', 'airline_claim_rate', 'airline_risk_bucket']],
    on='Airline',
    how='left'
)

df_with_airline_bucket[['Airline', 'airline_claim_rate', 'airline_risk_bucket']].head()

print("Data with Airline Risk Buckets:")
print(df_with_airline_bucket[['Airline', 'airline_claim_rate', 'airline_risk_bucket']].head(15))

#suggested pricing strategy based on airline risk bucket
#low risk: $15, medium risk: $35, high risk: $75
price_map = {
    "Low": 15,
    "Medium": 35,
    "High": 75
}

df_with_airline_bucket['suggested_price'] = df_with_airline_bucket['airline_risk_bucket'].map(price_map)
print("Suggested Pricing Strategy:")
print(df_with_airline_bucket[['Airline', 'airline_risk_bucket', 'suggested_price']].head(15))