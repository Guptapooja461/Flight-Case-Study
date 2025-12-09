import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("flight_reports/figures", exist_ok=True)

# Load cleaned data
df_cleaned_flights = pd.read_parquet('cleaned_flight_delays.parquet')
print(df_cleaned_flights.head())

# Exploratory Data Analysis (EDA)
#here is_claim has two values 0 and 800,which might create a lot of issue in statistics or model building,so we will convert 800 to 1 for better understanding
df_cleaned_flights['is_claim_flag'] = df_cleaned_flights[['is_claim']].applymap(lambda x: 1 if x == 800 else 0)
overall_claim_rate = df_cleaned_flights['is_claim_flag'].mean()
print(f"Overall Claim Rate: {overall_claim_rate:.2%}")

print(df_cleaned_flights['delay_time'].describe())


#hypothesis 1: airline higher claime rate than others
airline_claim_rates = (df_cleaned_flights.groupby('Airline')['is_claim_flag'].mean().reset_index().rename(columns={'is_claim_flag': 'claim_rate'}))
print("Airline Claim Rates:")
print(airline_claim_rates)


#hypothesis 2: hour higher claim rate than others
std_hour_claim_rates = (df_cleaned_flights.groupby('std_hour')['is_claim_flag'].mean().reset_index().rename(columns={'is_claim_flag': 'claim_rate'}))
print("Scheduled Hour Claim Rates:")
print(std_hour_claim_rates)

#hypthesis 3: month higher claim rate than others
month_claim_rates = (df_cleaned_flights.groupby('Month')['is_claim_flag'].mean().reset_index().rename(columns={'is_claim_flag': 'claim_rate'}))
print("Month Claim Rates:")
print(month_claim_rates)




####visualizations of overall claim rates
fig, ax = plt.subplots()
ax.bar(["Overall"], [overall_claim_rate])
ax.set_ylabel("Claim Rate")
ax.set_title("Overall Flight Delay Claim Rate")
plt.tight_layout()
plt.savefig("flight_reports/figures/overall_claim_rate.png", dpi=150)
plt.show()

####visualizations for hypothesis 1:top 10 airlines with highest claim rates
# Count number of flights per airline
airline_counts = df_cleaned_flights['Airline'].value_counts().reset_index()
airline_counts.columns = ['Airline', 'count']

# Combine claim rates with flight counts
airline_data = airline_claim_rates.merge(airline_counts, on='Airline')

# Filter airlines with more than 500 flights and get top 10
top_airlines = airline_data[airline_data['count'] > 500].head(10)

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(top_airlines['Airline'], top_airlines['claim_rate'])
ax.set_xticklabels(top_airlines['Airline'], rotation=45, ha='right')
ax.set_ylabel("Claim Rate")
ax.set_title("Claim Rate by Airline (Top by Volume)")
plt.tight_layout()
plt.savefig("flight_reports/figures/claim_rate_by_airline.png", dpi=150)
plt.show()


####visualizations for hypothesis 2:scheduled hour higher claim rate than others
fig, ax = plt.subplots()
ax.plot(std_hour_claim_rates['std_hour'], std_hour_claim_rates['claim_rate'], marker='o')
ax.set_xlabel("Scheduled Departure Hour")
ax.set_ylabel("Claim Rate")
ax.set_title("Claim Rate by Departure Hour")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("flight_reports/figures/claim_rate_by_hour.png", dpi=150)
plt.show()

####visualizations for hypothesis 3:month higher claim rate than others
fig, ax = plt.subplots()
ax.plot(month_claim_rates['Month'], month_claim_rates['claim_rate'], marker='o')
ax.set_xlabel("Month")
ax.set_ylabel("Claim Rate")
ax.set_title("Claim Rate by Month")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("flight_reports/figures/claim_rate_by_month.png", dpi=150)
plt.show()



#visualizations for known airlines vs unknown airline claim rates
known_airline_claim_rate = df_cleaned_flights[df_cleaned_flights['Airline'] != 'Unknown']['is_claim_flag'].mean()
unknown_airline_claim_rate = df_cleaned_flights[df_cleaned_flights['Airline'] == 'Unknown']['is_claim_flag'].mean() 
fig, ax = plt.subplots()
ax.bar(['Known Airlines', 'Unknown Airline'], [known_airline_claim_rate, unknown_airline_claim_rate], color=['blue', 'orange'])
ax.set_ylabel("Claim Rate")
ax.set_title("Claim Rate: Known vs Unknown Airlines")
plt.tight_layout()
plt.savefig("flight_reports/figures/known_vs_unknown_airline_claim_rate.png", dpi=150)
plt.show()