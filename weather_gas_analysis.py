import pandas as pd
import numpy as np
import yfinance as yf
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple

# setup
TTF_TICKER = "TTF=F"
AMS_LAT = 52.36
AMS_LON = 4.90
HDD_BASE = 18.0
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

def fetch_gas_data(start_date, end_date):
    """Download TTF front month from yfinance, return df with TTF_Price"""
    raw = yf.download(TTF_TICKER, start=start_date, end=end_date, progress=False)

    # yfinance gives multi-level columns ugh
    gas_prices = raw.Close.copy()
    gas_prices.columns = ['TTF_Price']

    # gotta make sure timezone matches weather data later
    if gas_prices.index.tz is None:
        gas_prices.index = gas_prices.index.tz_localize('UTC')
    else:
        gas_prices.index = gas_prices.index.tz_convert('UTC')

    # gas flows 24/7 but settlements dont happen on weekends
    # forward fill so we have prices for every calendar day
    gas_prices = gas_prices.asfreq('D').ffill()

    return gas_prices

def fetch_weather_data(start_date, end_date, latitude, longitude):
    """Pull daily avg temp from Open-Meteo Archive API, no key needed"""
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "UTC"
    }

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # dates come back as YYYY-MM-DD strings, parse them
    dates = data['daily']['time']
    temps = data['daily']['temperature_2m_mean']

    weather = pd.DataFrame({
        'Avg_Temp_C': temps
    }, index=pd.to_datetime(dates))

    # force to UTC so merge doesnt silently drop rows
    weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')

    return weather

def calculate_hdd(df, temp_col='Avg_Temp_C'):
    """Heating degree days = how much heating is needed. base 18c"""
    df = df.copy()
    df['HDD'] = (HDD_BASE - df[temp_col]).clip(lower=0)
    return df

def merge_data(gas_df, weather_df):
    """Join gas and weather on datetime index, keep only dates we have both"""
    # ffill gas first - weekends dont exist in physical gas markets
    gas_df['TTF_Price'] = gas_df['TTF_Price'].ffill()

    merged = gas_df.join(weather_df, how='inner')
    return merged

def run_regression_analysis(df, x_col='HDD', y_col='TTF_Price'):
    """OLS of price on HDD, returns df with Predicted_Price and Residuals"""
    X = sm.add_constant(df[x_col])
    y = df[y_col]

    model = sm.OLS(y, X).fit()
    print(model.summary())

    df = df.copy()
    df['Predicted_Price'] = model.fittedvalues
    df['Residuals'] = model.resid

    print(f"R-squared: {model.rsquared:.3f}")
    print(f"HDD coefficient: {model.params['HDD']:.2f} EUR/MWh per HDD")

    return df, model

def visualize_results(df):
    """Top: TTF price. Bottom: residuals + red outlier dots at 2sigma"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(df.index, df['TTF_Price'], color='orange', linewidth=1)
    ax1.set_ylabel('TTF Price (EUR/MWh)')
    ax1.set_title('TTF Gas Price')

    thresh = 2 * df['Residuals'].std()
    outliers = df[np.abs(df['Residuals']) > thresh]

    ax2.plot(df.index, df['Residuals'], color='gray', linewidth=0.8)
    ax2.scatter(outliers.index, outliers['Residuals'], color='red', s=15)
    ax2.axhline(thresh, color='red', ls='--', alpha=0.5)
    ax2.axhline(-thresh, color='red', ls='--', alpha=0.5)
    ax2.set_ylabel('Residuals (EUR/MWh)')
    ax2.set_title('Residuals')

    plt.tight_layout()
    plt.show()

# ---- run the thing ----

print("fetching TTF data...")
gas_df = fetch_gas_data(START_DATE, END_DATE)
print(f"  got {len(gas_df)} days of gas prices")

print("fetching weather data for Amsterdam...")
weather_df = fetch_weather_data(START_DATE, END_DATE, AMS_LAT, AMS_LON)
print(f"  got {len(weather_df)} days of temps")

print("calculating HDD...")
weather_df = calculate_hdd(weather_df)

print("merging datasets...")
merged_df = merge_data(gas_df, weather_df)
print(f"  merged to {len(merged_df)} rows")

print("running regression...")
merged_df, model = run_regression_analysis(merged_df)

print("plotting...")
visualize_results(merged_df)

print("done")
