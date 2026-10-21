import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
from pathlib import Path


# Settings
ticker = "NVDA"
start_date = "2025-01-01"
end_date = "2026-01-01"
forecast_days = 5
run_interval = 5
n_simulations = 100_000
training_years = 5
random_seed = 42

output_folder = Path("garch_backtest_results")
output_folder.mkdir(exist_ok=True)


# Download historical prices
print("Downloading NVDA price data...")

data = yf.download(
    ticker,
    start=f"{int(start_date[:4]) - training_years}-01-01",
    end=end_date,
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise ValueError("No price data was downloaded.")

# Handle the column format returned by different yfinance versions
if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"].iloc[:, 0]
else:
    prices = data["Close"]

prices = prices.dropna().astype(float)
prices.index = pd.to_datetime(prices.index).tz_localize(None)
prices = prices[~prices.index.duplicated(keep="last")].sort_index()

if len(prices) < 500:
    raise ValueError("Not enough historical prices to run the backtest.")


# Choose forecast dates, using actual trading days
all_dates = prices.index
start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date)

forecast_starts = [
    i for i, date in enumerate(all_dates)
    if start <= date < end and i + forecast_days < len(all_dates)
]
forecast_starts = forecast_starts[::run_interval]

if not forecast_starts:
    raise ValueError("No forecast dates were found.")

print(f"Backtest period: {start_date} to {end_date}")
print(f"Forecast runs: {len(forecast_starts)}")
print(f"Simulations per run: {n_simulations:,}\n")


# Run the backtest
results = []

for run, i in enumerate(forecast_starts, start=1):
    forecast_date = all_dates[i]
    end_forecast_date = all_dates[i + forecast_days]

    # Use the five years of prices available at the forecast date
    training_start = forecast_date - pd.DateOffset(years=training_years)
    training_prices = prices.loc[
        (prices.index >= training_start) &
        (prices.index <= forecast_date)
    ]

    returns = np.log(training_prices / training_prices.shift(1)).dropna() * 100

    if len(returns) < 100:
        print(f"Skipping {forecast_date:%Y-%m-%d}: not enough data.")
        continue

    # Fit a normal GARCH(1,1) model
    model = arch_model(
        returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False
    )
    fitted_model = model.fit(disp="off")

    if fitted_model.convergence_flag != 0:
        print(f"Skipping {forecast_date:%Y-%m-%d}: model did not converge.")
        continue

    omega = float(fitted_model.params["omega"])
    alpha = float(fitted_model.params["alpha[1]"])
    beta = float(fitted_model.params["beta[1]"])

    last_variance = float(fitted_model.conditional_volatility.iloc[-1] ** 2)
    last_return = float(fitted_model.resid.iloc[-1])

    start_price = float(prices.loc[forecast_date])
    actual_price = float(prices.loc[end_forecast_date])

    # Simulate possible prices over the next five trading days
    rng = np.random.default_rng(random_seed + run)
    shocks = rng.standard_normal((n_simulations, forecast_days))

    simulated_prices = np.full(n_simulations, start_price)
    variance = np.full(n_simulations, last_variance)
    previous_return = np.full(n_simulations, last_return)
    daily_volatility = []

    for day in range(forecast_days):
        variance = (
            omega
            + alpha * previous_return**2
            + beta * variance
        )
        variance = np.maximum(variance, 1e-12)

        daily_volatility.append(float(np.sqrt(np.mean(variance))))

        simulated_return = np.sqrt(variance) * shocks[:, day]
        simulated_prices *= np.exp(simulated_return / 100)
        previous_return = simulated_return

    median_price = float(np.percentile(simulated_prices, 50))
    mean_price = float(np.mean(simulated_prices))
    lower_price = float(np.percentile(simulated_prices, 5))
    upper_price = float(np.percentile(simulated_prices, 95))

    price_error = median_price - actual_price
    actual_return = np.log(actual_price / start_price) * 100
    forecast_return = np.log(median_price / start_price) * 100

    # Realised volatility over the same five-day period
    period_prices = prices.loc[forecast_date:end_forecast_date]
    period_returns = np.log(period_prices / period_prices.shift(1)).dropna() * 100
    actual_volatility = float(np.sqrt(np.sum(period_returns**2)))
    forecast_volatility = float(np.sqrt(np.sum(np.array(daily_volatility)**2)))

    results.append({
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "end_date": end_forecast_date.strftime("%Y-%m-%d"),
        "starting_price": start_price,
        "actual_price": actual_price,
        "median_forecast": median_price,
        "mean_forecast": mean_price,
        "5th_percentile": lower_price,
        "95th_percentile": upper_price,
        "price_error": price_error,
        "absolute_price_error": abs(price_error),
        "absolute_percentage_error": abs(price_error) / actual_price * 100,
        "actual_5_day_return_pct": actual_return,
        "forecast_5_day_return_pct": forecast_return,
        "actual_5_day_volatility_pct": actual_volatility,
        "forecast_5_day_volatility_pct": forecast_volatility,
        "absolute_volatility_error": abs(forecast_volatility - actual_volatility),
        "inside_90_percent_interval": lower_price <= actual_price <= upper_price,
        "interval_width": upper_price - lower_price
    })

    print(
        f"{run:02d}/{len(forecast_starts)}  "
        f"{forecast_date:%Y-%m-%d} to {end_forecast_date:%Y-%m-%d}  "
        f"Actual: ${actual_price:.2f}  "
        f"Median forecast: ${median_price:.2f}"
    )


# Save the results
if not results:
    raise RuntimeError("The backtest did not produce any results.")

results_df = pd.DataFrame(results)
results_df.to_csv(output_folder / "garch_backtest_results.csv", index=False)

summary = pd.DataFrame([{
    "model": "GARCH(1,1) normal",
    "forecast_runs": len(results_df),
    "simulations_per_run": n_simulations,
    "median_absolute_price_error": results_df["absolute_price_error"].median(),
    "mean_absolute_price_error": results_df["absolute_price_error"].mean(),
    "root_mean_squared_price_error": np.sqrt(np.mean(results_df["price_error"]**2)),
    "mean_absolute_percentage_error_pct": results_df["absolute_percentage_error"].mean(),
    "90_percent_interval_coverage_pct": (
        results_df["inside_90_percent_interval"].mean() * 100
    ),
    "average_interval_width": results_df["interval_width"].mean(),
    "mean_absolute_volatility_error": results_df["absolute_volatility_error"].mean()
}])

summary.to_csv(output_folder / "backtestSummary.csv", index=False)

# Plot actual prices and the forecast ranges
x = np.arange(len(results_df))

plt.figure(figsize=(12, 6))
plt.plot(
    x,
    results_df["actual_price"],
    marker="o",
    label="Actual price"
)
plt.plot(
    x,
    results_df["median_forecast"],
    marker="o",
    label="Median forecast"
)
plt.fill_between(
    x,
    results_df["5th_percentile"].to_numpy(),
    results_df["95th_percentile"].to_numpy(),
    alpha=0.2,
    label="5th-95th percentile range"
)
plt.xticks(x, results_df["end_date"], rotation=60)
plt.title("NVDA GARCH(1,1) Backtest")
plt.xlabel("Forecast end date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig(output_folder / "garch_backtest_prices.png", dpi=200)
plt.show()


# Compare forecast and realised volatility
plt.figure(figsize=(12, 6))
plt.plot(
    results_df["end_date"],
    results_df["actual_5_day_volatility_pct"],
    marker="o",
    label="Realised volatility"
)
plt.plot(
    results_df["end_date"],
    results_df["forecast_5_day_volatility_pct"],
    marker="o",
    label="Forecast volatility"
)
plt.xticks(rotation=60)
plt.title("Five-Day Volatility: Forecast vs Actual")
plt.xlabel("Forecast end date")
plt.ylabel("Volatility (%)")
plt.legend()
plt.tight_layout()
plt.savefig(output_folder / "garch_backtest_volatility.png", dpi=200)
plt.show()


# Plot the difference between the median forecast and the actual price
plt.figure(figsize=(12, 5))
plt.axhline(0, linewidth=1)
plt.bar(results_df["end_date"], results_df["price_error"])
plt.xticks(rotation=60)
plt.title("GARCH Median Forecast Errors")
plt.xlabel("Forecast end date")
plt.ylabel("Forecast price - actual price ($)")
plt.tight_layout()
plt.savefig(output_folder / "garch_backtest_errors.png", dpi=200)
plt.show()
