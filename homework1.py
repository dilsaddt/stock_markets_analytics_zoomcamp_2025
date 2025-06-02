# First homework here: https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2025/homework%201.md
# /Users/dagtekdi/Dropbox/dilsad_work/SMAZ_2025

#################################
########## QUESTION 1 ###########
#################################

import pandas as pd
from datetime import datetime

from numpy.array_api import from_dlpack

# Load S&P 500 data from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
df = tables[0]  # First table contains the data we need
df.info() # str in R

# df = df[['Symbol', 'Security', 'Date added']] same with below
df = df.iloc[:, [0, 1, 5]]
df.info() # str in R

# Rename columns for easier access if needed
# df = df.rename(columns={"Date first added": "Date Added"})

# Extract year from 'Date Added'
df["Year Added"] = pd.to_datetime(df["Date added"], errors='coerce').dt.year

# Drop NaN years (companies without a recorded 'Date Added')
year_counts = df["Year Added"].dropna().astype(int)

# Filter out 1957, the year S&P 500 started
year_counts_filtered = year_counts[year_counts != 1957]

# Count how many additions per year
additions_per_year = year_counts_filtered.value_counts()

# Find the year with the highest number of additions (latest if tie)
most_additions = additions_per_year[additions_per_year == additions_per_year.max()]
most_additions_year = most_additions.index.max()

print("Year with the highest number of additions (excluding 1957):", most_additions_year)
print("\nAdditions per year:\n", additions_per_year.sort_values(ascending=False).head())

# Answer: 2017 (and 2016)

# Part 2: How many current stocks have been in the index more than 20 years?
today = datetime.today()
df["Years in Index"] = today.year - df["Year Added"]
long_term_members = df[df["Years in Index"] > 20]
print("\nNumber of companies in the S&P 500 for more than 20 years:", len(long_term_members))

# Answer: Number of companies in the S&P 500 for more than 20 years: 219

#################################
########## QUESTION 2 ###########
#################################
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the indices
indices = {
    'US (S&P 500)': '^GSPC',
    'China (Shanghai)': '000001.SS',
    'Hong Kong (Hang Seng)': '^HSI',
    'Australia (ASX 200)': '^AXJO',
    'India (Nifty 50)': '^NSEI',
    'Canada (TSX)': '^GSPTSE',
    'Germany (DAX)': '^GDAXI',
    'UK (FTSE 100)': '^FTSE',
    'Japan (Nikkei 225)': '^N225',
    'Mexico (IPC)': '^MXX',
    'Brazil (Ibovespa)': '^BVSP'
}

start_date = '2025-01-01'
end_date = '2025-05-01'

# Use a list to store results
returns_list = []

# Download data and calculate YTD returns
for name, ticker in indices.items():
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    plt.plot(data['Close'], label=name, marker='o')
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    ytd_return = float(((end_price - start_price) / start_price) * 100)
    returns_list.append([name, ytd_return])
plt.legend()

# Convert to DataFrame
returns_df = pd.DataFrame(returns_list, columns=['Index', 'YTD Return (%)'])
returns_df = returns_df.dropna().sort_values(by='YTD Return (%)', ascending=False)

## Two solutions:
# First: find the row number of US (S&P 500), until that row number that is how many countries with larger YTD returns
# because the data frame is ordered
returns_df.index.get_loc(returns_df[returns_df['Index'] == 'US (S&P 500)'].index[0]) + 1 # +1 because it starts with 0
# US is at 10th row, so 9 countries are larger

# Second: Check who has larger YTD returns and count them
# Get the return of the US (S&P 500)
sp500_return = returns_df[returns_df['Index'] == 'US (S&P 500)']['YTD Return (%)'].values[0]

# Filter the DataFrame for returns greater than S&P 500
better_than_sp500 = returns_df[returns_df['YTD Return (%)'] > sp500_return]
print(better_than_sp500)
len(better_than_sp500) # 9

# Answer: 9 countries

plt.plot(returns_df['YTD Return (%)'])
# plt.plot(returns_df.index, returns_df['YTD Return (%)'])
# plt.show()

#################################
########## QUESTION 3 ###########
#################################

import yfinance as yf
import pandas as pd

# Step 1: Download S&P 500 historical data (1950-present) using yfinance
sp500 = yf.download("^GSPC", start="1950-01-01", progress=False)[['Close']].dropna()
sp500.columns = ['_'.join(col).strip() for col in sp500.columns.values] # get rid of multi index

sp500_df = sp500.reset_index() # turn rownames date to column date, in another data frame
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
sp500_df = sp500_df.rename(columns={'Date': 'Date', 'Close_^GSPC': 'Close'}) # rename columns

# Step 2: Identify all-time high points (where price exceeds all previous prices)
sp500_df['all_time_high'] = sp500_df['Close'].cummax()
sp500_df['new_high'] = sp500_df['Close'] == sp500_df['all_time_high']
highs = sp500_df[sp500_df['new_high']]

# Step 3: For each pair of consecutive all-time highs, find the minimum price in between
# Step 4: Calculate drawdown percentages: (high - low) / high × 100
# Step 5: Filter for corrections with at least 5% drawdown
# Step 6: Calculate the duration in days for each correction period

corrections = [] # empty list

for i in range(1, len(highs)):
    start = highs.iloc[i - 1]['Date']
    end = highs.iloc[i]['Date']
    between = sp500_df[(sp500_df['Date'] >= start) & (sp500_df['Date'] <= end)]

    if between.empty or len(between) < 2: # skip if there a no values in between or steps are less than 2
        continue

    min_price = between['Close'].min()
    min_row = between[between['Close'] == min_price].iloc[0]
    min_date = min_row['Date']

    peak_price = between['Close'].iloc[0]
    drawdown = (peak_price - min_price) / peak_price * 100
    duration = (min_date - start).days

    if drawdown >= 5: # we need corrections more tha 5%
        corrections.append({
            'Start': start,
            'Low': min_date,
            'End': end,
            'Drawdown (%)': round(drawdown, 2),
            'Duration (days)': duration
        })

df_corrections = pd.DataFrame(corrections)

# Step 7: Determine the 25th, 50th (median), and 75th percentiles for correction durations
percentiles = df_corrections['Duration (days)'].quantile([0.25, 0.5, 0.75])

# Results
df_sorted = df_corrections.sort_values(by='Drawdown (%)', ascending=False) # sort by largest drawdowns

# Format each row into the desired string
for _, row in df_sorted.iterrows():
    start = pd.to_datetime(row['Start']).strftime('%Y-%m-%d')
    end = pd.to_datetime(row['End']).strftime('%Y-%m-%d')
    drawdown = round(row['Drawdown (%)'], 1)
    duration = int(row['Duration (days)'])

    print(f"{start} to {end}: {drawdown}% drawdown over {duration} days")

# top 10
top10 = df_corrections.sort_values(by='Drawdown (%)', ascending=False).head(10)

for _, row in top10.iterrows():
    start = pd.to_datetime(row['Start']).strftime('%Y-%m-%d')
    end = pd.to_datetime(row['End']).strftime('%Y-%m-%d')
    drawdown = round(row['Drawdown (%)'], 1)
    duration = int(row['Duration (days)'])

    print(f"{start} to {end}: {drawdown}% drawdown over {duration} days")

# The median duration
percentiles[0.5] # median
# Answer: 39.0

#################################
########## QUESTION 4 ###########
#################################

import pandas as pd
import yfinance as yf
import numpy as np

# Step 1: Load earnings data from CSV (ha1_Amazon.csv) containing earnings dates, EPS estimates, and actual EPS. Make sure you are using the correct delimiter to read the data, such as in this command python pandas.read_csv("ha1_Amazon.csv", delimiter=';')

amzn_earnings = pd.read_csv("/Users/dagtekdi/Dropbox/dilsad_work/SMAZ_2025/ha1_Amazon.csv", delimiter=';')

# Step 2: Download complete historical price data using yfinance
amzn_price = yf.download('AMZN', progress=False)
amzn_price.columns = ['_'.join(col).strip() for col in amzn_price.columns.values] # get rid of multi index
amzn_price = amzn_price.reset_index() # turn rownames date to column date, in another data frame
amzn_price['Date'] = pd.to_datetime(amzn_price['Date'])

# Step 3: Calculate 2-day percentage changes for all historical dates: for each sequence of 3 consecutive trading days
# (Day 1, Day 2, Day 3), compute the return as Close_Day3 / Close_Day1 - 1. (Assume Day 2 may correspond to the earnings announcement.)
amzn_price['2d_perc'] = (amzn_price['Close_AMZN'].shift(-2) / amzn_price['Close_AMZN']) - 1

# Step 4: Identify positive earnings surprises (where "actual EPS > estimated EPS" OR "Surprise (%)>0")
amzn_earnings['Surprise (%)'] = pd.to_numeric(
    amzn_earnings['Surprise (%)'], errors='coerce'
)
amzn_earnings['Surprise Positive'] = (
    (amzn_earnings['Surprise (%)'] > 0)
)

### OR ###

#
# amzn_earnings['Surprise Positive'] = (
#     (amzn_earnings['Reported EPS'] > amzn_earnings['EPS Estimate'])
# )


# Step 5: Calculate 2-day percentage changes following positive earnings surprises.
# Show your answer in % (closest number to the 2nd digit): return * 100.0
# Get 2-day returns after positive surprises (as percentages)

# fix the date to match with the one in amzn_price
amzn_earnings['Date'] = pd.to_datetime(amzn_earnings['Earnings Date'].str.replace(r'\s+at.*(EDT|EST)', '', regex=True),errors='coerce')

positive_dates = amzn_earnings.loc[amzn_earnings['Surprise Positive'] == True, 'Date']
matching_indices = amzn_price.index[amzn_price['Date'].isin(positive_dates)].tolist()

positive_earning_2dp = amzn_price['2d_perc'].iloc[matching_indices]
positive_earning_2dp = positive_earning_2dp.dropna()
positive_earning_2dp
positive_earning_2dp.median()
positive_earning_2dp.median()*100

# positive_earning_2dp_perc = positive_earning_2dp.dropna()*100
# positive_earning_2dp_perc = positive_earning_2dp_perc.dropna().round(2)
# positive_earning_2dp_perc
# positive_earning_2dp_perc.median()


# get which dates have positive surprise
pos_surp_dates = amzn_earnings.loc[amzn_earnings['Surprise Positive'] == True, 'Date']

# filter historical data acc to these dates
amzn_price_pos_surp = amzn_price[amzn_price['Date'].isin(pos_surp_dates)]

# now calculate 2-day returns again but in percentage
amzn_price_pos_surp['2d_perc_after_surp'] = (
       amzn_price_pos_surp['Close_AMZN'].shift(-2) / (amzn_price_pos_surp['Close_AMZN'] - 1)
)

amzn_price_pos_surp['2d_perc_after_surp (%)'] = (
       (amzn_price_pos_surp['Close_AMZN'].shift(-2) / (amzn_price_pos_surp['Close_AMZN'] - 1))*100
)
# round to 2 decimal
amzn_price_pos_surp['2d_perc_after_surp (%)'] = amzn_price_pos_surp['2d_perc_after_surp (%)'].dropna().round(2)

# Step 6: (Optional) Compare the median 2-day percentage change for positive surprises vs. all historical dates. Do you see the difference
historical_returns = amzn_price['2d_perc'].median()
historical_returns = round(historical_returns,2)
historical_returns

pos_surp_returns = amzn_price_pos_surp['2d_perc_after_surp'].median()
pos_surp_returns = round(pos_surp_returns,2)
pos_surp_returns
