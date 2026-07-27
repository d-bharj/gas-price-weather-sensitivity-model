import pandas as pd
import numpy as np
import yfinance as yf
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TTF_TICKER = "TTF=F"
AMS_LAT = 52.36
AMS_LON = 4.90
HDD_BASE = 18.0          # EU standard base temp for HDD
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120


def fetch_gas_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Download TTF front-month futures from Yahoo Finance.

    Returns a daily DataFrame with a single column 'TTF_Price',
    forward-filled to cover non-settlement days (weekends, holidays).
    Timezone is normalised to UTC for consistent merging with weather data.
    """
    raw = yf.download(TTF_TICKER, start=start_date, end=end_date, progress=False)
    # yfinance returns MultiIndex on columns, extract  close series
    gas_prices = raw.Close.copy()
    gas_prices.columns = ['TTF_Price']
    # Normalise timezone — mismatched tz causes row-drops on join
    if gas_prices.index.tz is None:
        gas_prices.index = gas_prices.index.tz_localize('UTC')
    else:
        gas_prices.index = gas_prices.index.tz_convert('UTC')
    # gas flows on non-settlement days forward-fill gives an
    #operational (NOT trading) view of price for each calendar day
    gas_prices = gas_prices.asfreq('D').ffill()
    return gas_prices


def fetch_weather_data(start_date: str, end_date: str,
                       latitude: float, longitude: float) -> pd.DataFrame:
    """Fetch daily mean temperature from Open-Meteo Archive API.

    Uses Amsterdam (52.36N, 4.90E) as a weather proxy for the TTF trading hub.
    Open-Meteo requires no API key and provides ERA5-reanalysis data.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "UTC",
    }
    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    weather = pd.DataFrame(
        {'Avg_Temp_C': data['daily']['temperature_2m_mean']},
        index=pd.to_datetime(data['daily']['time']),
    )
    weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')
    return weather


# ---------------------------------------------------------------------------
# fetch, HDD, merge, OLS, visualise, export
# requires only:
# yfinance, requests, statsmodels, and matplotlib.
# ---------------------------------------------------------------------------

print("fetching TTF data...")
gas = fetch_gas_data(START_DATE, END_DATE)
print(f"  got {len(gas)} days of gas prices")

print("fetching weather data for Amsterdam...")
weather = fetch_weather_data(START_DATE, END_DATE, AMS_LAT, AMS_LON)
print(f"  got {len(weather)} days of temps")

#Step 1: Get Heating Degree Days (HDD)
# HDD = max(0, 18C - T_avg) the 18c base is EU standard: represents
# the outdoor temperature below which buildings require heating.
weather['HDD'] = (HDD_BASE - weather['Avg_Temp_C']).clip(lower=0)

# Step 2: Merge gas and weather on date
#  join retains only dates where both a settlement price and a
# temperature record exist. mismatched calendars excluded
gas['TTF_Price'] = gas['TTF_Price'].ffill()
df = gas.join(weather, how='inner')
print(f"  merged to {len(df)} rows")

# Step 3: OLS regression — TTF_Price ~ HDD
# Model: TTF_Price = beta_0 + beta_1 * HDD + epsilon
# beta_1 estimates marginal price impact of one additional HDD (EUR/MWh).
# the R-squared quantifies how much of price variance is explained by cold.
X = sm.add_constant(df['HDD'])
y = df['TTF_Price']
model = sm.OLS(y, X).fit()
print(model.summary())

df['Predicted_Price'] = model.fittedvalues
df['Residuals'] = model.resid
print(f"R-squared: {model.rsquared:.3f}")
print(f"HDD coefficient: {model.params['HDD']:.2f} EUR/MWh per HDD")

#Step 4: Visualise — price series, residual outliers
# residuals beyond 2 standard deviations correspond to supply-side
# events (e.g outages) rather than weather.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

ax1.plot(df.index, df['TTF_Price'], color='orange', linewidth=1)
ax1.set_ylabel('TTF Price (EUR/MWh)')
ax1.set_title('TTF Gas Price — Daily Front Month')

thresh = 2 * df['Residuals'].std()
outliers = df[np.abs(df['Residuals']) > thresh]

ax2.plot(df.index, df['Residuals'], color='gray', linewidth=0.8)
ax2.scatter(outliers.index, outliers['Residuals'], color='red', s=15)
ax2.axhline(thresh, color='red', ls='--', alpha=0.5)
ax2.axhline(-thresh, color='red', ls='--', alpha=0.5)
ax2.set_ylabel('Residuals (EUR/MWh)')
ax2.set_title(f'Residuals — {len(outliers)} outliers beyond {thresh:.1f} EUR/MWh')

plt.tight_layout()
plt.show()

#Step 5: Export for tableau
df.to_csv("ttf_analysis_output.csv", date_format="%Y-%m-%d")
print("exported to ttf_analysis_output.csv")
print("done")
