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


# File locations
project_folder = Path(__file__).resolve().parent.parent
results_folder = project_folder / "garch_backtest_results"
summary_file = results_folder / "backtestSummary.csv"

if not results_folder.exists():
    raise FileNotFoundError(
        f"Results folder not found: {results_folder}"
    )


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
    raise ValueError("No NVDA price data was downloaded.")

if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"].iloc[:, 0]
else:
    prices = data["Close"]

prices = prices.dropna().astype(float)
prices.index = pd.to_datetime(prices.index).tz_localize(None)
prices = prices[~prices.index.duplicated(keep="last")].sort_index()


# Set the forecast dates
start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date)

forecast_starts = [
    i for i, date in enumerate(prices.index)
    if start <= date < end and i + forecast_days < len(prices)
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
    forecast_date = prices.index[i]
    forecast_end = prices.index[i + forecast_days]

    # Use only prices available on the forecast date
    training_start = forecast_date - pd.DateOffset(years=training_years)

    training_prices = prices.loc[
        (prices.index >= training_start) &
        (prices.index <= forecast_date)
    ]

    returns = np.log(
        training_prices / training_prices.shift(1)
    ).dropna() * 100

    if len(returns) < 100:
        print(f"Skipping {forecast_date.date()}: not enough data.")
        continue

    # Fit GJR-GARCH(1,1) with t-distributed errors
    model = arch_model(
        returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="t",
        rescale=False
    )

    fitted = model.fit(disp="off")

    if fitted.convergence_flag != 0:
        print(f"Skipping {forecast_date.date()}: model did not converge.")
        continue

    omega = float(fitted.params["omega"])
    alpha = float(fitted.params["alpha[1]"])
    gamma = float(fitted.params["gamma[1]"])
    beta = float(fitted.params["beta[1]"])
    nu = float(fitted.params["nu"])

    if nu <= 2:
        print(f"Skipping {forecast_date.date()}: invalid t parameters.")
        continue

    last_variance = float(
        fitted.conditional_volatility.iloc[-1] ** 2
    )
    last_return = float(fitted.resid.iloc[-1])

    start_price = float(prices.iloc[i])
    actual_price = float(prices.iloc[i + forecast_days])

    # Generate standardised t-distributed shocks
    rng = np.random.default_rng(42 + run)

    shocks = (
        rng.standard_t(nu, size=(n_simulations, forecast_days))
        / np.sqrt(nu / (nu - 2))
    )

    simulated_prices = np.full(n_simulations, start_price)
    variance = np.full(n_simulations, last_variance)
    previous_return = np.full(n_simulations, last_return)
    daily_volatility = []

    for day in range(forecast_days):
        # Negative returns have an additional effect on volatility
        negative_return = (previous_return < 0).astype(float)

        variance = (
            omega
            + alpha * previous_return**2
            + gamma * negative_return * previous_return**2
            + beta * variance
        )

        variance = np.maximum(variance, 1e-12)

        daily_volatility.append(
            float(np.sqrt(np.mean(variance)))
        )

        simulated_return = np.sqrt(variance) * shocks[:, day]

        # Update the simulated share prices
        simulated_prices *= np.exp(
            simulated_return / 100 - 0.5 * variance / 10000
        )

        previous_return = simulated_return

    # Calculate forecast prices
    median_price = float(np.percentile(simulated_prices, 50))
    mean_price = float(np.mean(simulated_prices))
    lower_price = float(np.percentile(simulated_prices, 5))
    upper_price = float(np.percentile(simulated_prices, 95))

    price_error = median_price - actual_price

    actual_return = np.log(actual_price / start_price) * 100
    forecast_return = np.log(median_price / start_price) * 100

    # Calculate realised five-day volatility
    period_prices = prices.loc[forecast_date:forecast_end]

    period_returns = np.log(
        period_prices / period_prices.shift(1)
    ).dropna() * 100

    actual_volatility = float(np.sqrt(np.sum(period_returns**2)))
    forecast_volatility = float(
        np.sqrt(np.sum(np.array(daily_volatility)**2))
    )

    results.append({
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "forecast_end_date": forecast_end.strftime("%Y-%m-%d"),
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
        "absolute_volatility_error": abs(
            forecast_volatility - actual_volatility
        ),
        "inside_90_percent_interval": (
            lower_price <= actual_price <= upper_price
        ),
        "interval_width": upper_price - lower_price
    })

    print(
        f"{run:02d}/{len(forecast_starts)} "
        f"{forecast_date:%Y-%m-%d} to {forecast_end:%Y-%m-%d} | "
        f"Actual: ${actual_price:.2f} | "
        f"Forecast: ${median_price:.2f}"
    )


# Save detailed results
if not results:
    raise RuntimeError("No successful forecasts were produced.")

results_df = pd.DataFrame(results)

results_df.to_csv(
    results_folder / "GJRgarch-tBacktest_results.csv",
    index=False
)


# Calculate the summary metrics
summary = pd.DataFrame([{
    "model": "GJR-GARCH-t",
    "forecast_runs": len(results_df),
    "simulations_per_run": n_simulations,
    "median_absolute_price_error": (
        results_df["absolute_price_error"].median()
    ),
    "mean_absolute_price_error": (
        results_df["absolute_price_error"].mean()
    ),
    "root_mean_squared_price_error": np.sqrt(
        np.mean(results_df["price_error"]**2)
    ),
    "mean_absolute_percentage_error_pct": (
        results_df["absolute_percentage_error"].mean()
    ),
    "90_percent_interval_coverage_pct": (
        results_df["inside_90_percent_interval"].mean() * 100
    ),
    "average_interval_width": results_df["interval_width"].mean(),
    "mean_absolute_volatility_error": (
        results_df["absolute_volatility_error"].mean()
    )
}])


# Add this model to the existing summary
summary_columns = [
    "model",
    "forecast_runs",
    "simulations_per_run",
    "median_absolute_price_error",
    "mean_absolute_price_error",
    "root_mean_squared_price_error",
    "mean_absolute_percentage_error_pct",
    "90_percent_interval_coverage_pct",
    "average_interval_width",
    "mean_absolute_volatility_error"
]

if summary_file.exists():
    old_summary = pd.read_csv(summary_file)

    for column in summary_columns:
        if column not in old_summary.columns:
            old_summary[column] = np.nan

    old_summary = old_summary[summary_columns]
    old_summary = old_summary[
        old_summary["model"] != "GJR-GARCH-t"
    ]

    combined_summary = pd.concat(
        [old_summary, summary[summary_columns]],
        ignore_index=True
    )
else:
    combined_summary = summary[summary_columns]

combined_summary.to_csv(summary_file, index=False)

# Plot actual prices and forecast ranges
x = np.arange(len(results_df))

plt.figure(figsize=(12, 6))
plt.plot(x, results_df["actual_price"], marker="o", label="Actual price")
plt.plot(x, results_df["median_forecast"], marker="o", label="Median forecast")
plt.fill_between(
    x,
    results_df["5th_percentile"].to_numpy(),
    results_df["95th_percentile"].to_numpy(),
    alpha=0.2,
    label="5th-95th percentile range"
)
plt.xticks(x, results_df["forecast_end_date"], rotation=60)
plt.title("NVDA GJR-GARCH-t Backtest")
plt.xlabel("Forecast end date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig(
    results_folder / "GJRgarch-tBacktest_prices.png",
    dpi=200
)
plt.show()


# Compare forecast and realised volatility
plt.figure(figsize=(12, 6))
plt.plot(
    results_df["forecast_end_date"],
    results_df["actual_5_day_volatility_pct"],
    marker="o",
    label="Realised volatility"
)
plt.plot(
    results_df["forecast_end_date"],
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
plt.savefig(
    results_folder / "GJRgarch-tBacktest_volatility.png",
    dpi=200
)
plt.show()


# Plot forecast errors
plt.figure(figsize=(12, 5))
plt.axhline(0, linewidth=1)
plt.bar(results_df["forecast_end_date"], results_df["price_error"])
plt.xticks(rotation=60)
plt.title("GJR-GARCH-t Forecast Errors")
plt.xlabel("Forecast end date")
plt.ylabel("Forecast price - actual price ($)")
plt.tight_layout()
plt.savefig(
    results_folder / "GJRgarch-tBacktest_errors.png",
    dpi=200
)
plt.show()
