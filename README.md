# NVDA Volatility Forecasting

## Overview

This project compares three GARCH-based models for forecasting stock price movements over a five-trading-day period. Monte Carlo simulation is used to generate possible future prices, with the models compared using historical backtesting with NVIDIA (NVDA) being used as a case study.

The aim is to see whether adding a t distribution, asymmetric volatility and implied volatility improves the forecasts.


## Models

- **GARCH(1,1):** Uses past returns and volatility to estimate future volatility, with normally distributed shocks.
- **GARCH(1,1)-t:** Uses a t distribution to allow for larger return movements.
- **GJR-GARCH(1,1)-t-IV:** Accounts for the different effects of positive and negative returns and incorporates options-implied volatility.

## Mathematical Methods

Daily log returns are calculated as:

\[
r_t=\ln\left(\frac{S_t}{S_{t-1}}\right)
\]

The standard GARCH model estimates conditional variance using:

\[
h_t=\omega+\alpha\epsilon_{t-1}^2+\beta h_{t-1}
\]

The GJR-GARCH model adds a term to account for negative returns:

\[
h_t=\omega+\alpha\epsilon_{t-1}^2+
\gamma I(\epsilon_{t-1}<0)\epsilon_{t-1}^2+\beta h_{t-1}
\]

For the IV model, GJR-GARCH variance and implied variance are combined using equal weights:

\[
h_t^{blend}=0.5h_t^{GJR}+0.5h_t^{IV}
\]

Monte Carlo simulation then generates 100,000 possible price paths over five trading days.

## Backtesting

The models are tested across 49 historical forecast dates. Performance is measured using:

- Median and mean absolute price error
- Root mean squared error (RMSE)
- Mean absolute percentage error (MAPE)
- 90% prediction interval coverage
- Average prediction interval width
- Mean absolute volatility error

## Tools and Data

- Python
- NumPy, pandas and SciPy
- `arch` for GARCH model estimation
- yfinance for historical prices and options data
