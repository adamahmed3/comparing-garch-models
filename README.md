# NVDA Volatility Forecasting

## Overview

This project compares three GARCH-based models for forecasting stock price movements over a five-trading-day period. Monte Carlo simulation is used to generate possible future prices, with the models compared using historical backtesting with NVIDIA (NVDA) being used as a case study.

The aim is to see whether adding a t distribution, asymmetric volatility and implied volatility improves the forecasts.

## Models

- **GARCH(1,1):** Uses normally distributed shocks to estimate future volatility based on past returns and volatility.
- **GJR-GARCH(1,1)-t:** Adjusts for the different effects of positive and negative returns, using a t distribution for more probability of large shocks.
- **GJR-GARCH(1,1)-t-IV:** Builds on GJR-GARCH-t by incorporating options-implied volatility into the variance estimate.

## Mathematical Methods

### 1. Log Returns

Daily log returns are calculated from closing prices:

$$
r_t=\ln\left(\frac{S_t}{S_{t-1}}\right)
$$

then multiplied by 100 for model estimation.

### 2. GARCH(1,1)

The standard normal GARCH model estimates conditional variance using the previous period's squared error and variance:

$$
h_t=\omega+\alpha\epsilon_{t-1}^2+\beta h_{t-1}
$$

\(\omega\) is the variance constant, \(\alpha\) measures the effect of recent shocks, and \(\beta\) measures the persistence of volatility.

### 3. GJR-GARCH(1,1)-t

The GJR-GARCH model accounts for the effect of negative returns, allowing negative shocks to affect volatility differently from positive shocks.  The t distribution makes larger shocks more likely. This makes larger price movements more likely in the simulations, helping the model account for extreme market movements.

$$
\begin{aligned}
h_t={}&\omega+\alpha\epsilon_{t-1}^2\\
&+\gamma I(\epsilon_{t-1}<0)\epsilon_{t-1}^2\\
&+\beta h_{t-1}
\end{aligned}
$$

The indicator function \(I(\epsilon_{t-1}<0)\) equals 1 when the previous shock is negative and 0 otherwise. The model uses a t distribution for the return shocks.

### 4. Incorporating Implied Volatility

The third model combines GJR-GARCH-t variance with options-implied variance using equal weighting:

$$
h_t^{blend}=0.5h_t^{GJR}+0.5h_t^{IV}
$$

This allows the model to use both historical price behaviour and information from the options market (for backtesting, the historical data was taken from bloomberg terminal).

## Monte Carlo Simulation

Each model is used to simulate 100,000 possible price paths over five trading days. Simulated returns are added to the latest price to generate possible future NVDA prices.

The price update is:

$$
S_{t+1}=S_t\exp\left(\frac{\epsilon_{t+1}}{100}\right)
$$

The resulting paths are used to examine the range of possible future prices to estimate prediction intervals.

## Backtesting

The models are evaluated across 49 historical forecast dates. Their forecasts are compared with actual prices using the following metrics:

- **Median absolute price error:** The median absolute difference between forecast and actual prices.
- **Mean absolute error (MAE):** The average absolute price error.
- **Root mean squared error (RMSE):** Gives greater weight to larger forecast errors.
- **Mean absolute percentage error (MAPE):** Measures average error as a percentage of actual prices.
- **90% prediction interval coverage:** The percentage of actual prices falling within the predicted intervals.
- **Average interval width:** Measures how wide the prediction intervals are.
- **Mean absolute volatility error:** Measures the difference between forecast and realised volatility.

These metrics help compare price accuracy, volatility forecasts and the uncertainty captured by each model.

## Tools and Data

- Python
- NumPy and pandas for data handling
- SciPy for statistical calculations
- `arch` for GARCH model estimation
- yfinance for historical prices and options data
- Bloomberg historical implied-volatility data for the IV backtest
- Matplotlib for visualising the results



This project is for educational purposes and is not investment advice.
