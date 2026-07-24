# Weather Anomaly & European Gas Price Analysis

## The Problem

Gas prices move on weather and on supply shocks
This script tries to split those apart using Heating Degree Days and a simple regression

## The Solution

Pulls TTF prices from Yahoo Finance and Amsterdam temps from Open-Meteo
Calculates HDD how much heating is needed based on 18 degree base
Runs an OLS regression of price on HDD
Residuals that are way off (2 standard deviations) are probably supply shocks not weather

## Key Decisions

- HDD base set to 18 degrees which is the standard EU convention
- Amsterdam weather as proxy for the TTF hub since thats where TTF physically delivers
- TTF=F continuous front month from Yahoo Finance free no API key
- Pure OLS regression for simplicity non-linear models could do better
- Date range 2022-01-01 to 2025-12-31 covers the crisis and normalisation

## Challenges Encountered

- YFinance returns multi-level columns had to flatten to get Close prices
- Open-Meteo returns dates as strings had to parse them
- One dataset was timezone naive the other was UTC aware merging silently dropped everything until I forced both to UTC
- Gas trades 24/7 but prices only settle on business days forward filled weekends
- TTF=F history on Yahoo is limited only goes back to like 2017
- Open-Meteo has a 5 day lag before data shows up

## Future Improvements

- Non-linear models gradient boosting or splines for saturation effects
- Rolling regression to see how weather sensitivity changes over time
- Regime switching to separate crisis periods from normal markets
- Multiple cities weighted by gas demand instead of just Amsterdam
- Storage data and pipeline flows would explain a lot more of the residuals
- Use the full forward curve instead of just front month

## Tableau Export

A CSV export line has been added to the script `merged_df.to_csv("ttf_analysis_output.csv")` which outputs all columns TTF_Price Avg_Temp_C HDD Predicted_Price and Residuals for direct use in Tableau dashboards
