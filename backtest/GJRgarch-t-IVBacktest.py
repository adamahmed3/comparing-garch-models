from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model


# Settings
ticker = "NVDA"

backtest_start = "2025-01-01"
backtest_end = "2026-01-01"

training_years = 5
forecast_days = 5
forecast_step = 5
n_simulations = 100_000

# Same as w_garch in the live forecast script
garch_weight = 0.5
random_seed = 42


# File locations
script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent

iv_file = project_folder / "30DAYIv(Sheet1).csv"

results_folder = project_folder / "garch_backtest_results"
results_folder.mkdir(parents=True, exist_ok=True)

results_file = results_folder / "GJRgarch-t-ivBacktest_results.csv"
summary_file = results_folder / "backtestSummary.csv"

price_plot_file = results_folder / "GJRgarch-t-ivBacktest_prices.png"
vol_plot_file = results_folder / "GJRgarch-t-ivBacktest_volatility.png"
error_plot_file = results_folder / "GJRgarch-t-ivBacktest_errors.png"


# Load the Bloomberg IV data
def load_iv(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the IV file: {file_path}"
        )

    # Skip the Bloomberg metadata rows, the real header is on line 6
    data = pd.read_csv(file_path, skiprows=5)
    data.columns = [
        str(column).strip().replace("\ufeff", "")
        for column in data.columns
    ]

    if data.empty:
        raise ValueError("The IV file is empty.")

    # Columns are "Dates" and "30DAY_IMPVOL_100.0%MNY_DF"
    date_column = data.columns[0]
    iv_column = data.columns[1]

    # Dates are day-first (dd/mm/yyyy)
    data[date_column] = pd.to_datetime(
        data[date_column], format="%d/%m/%Y", errors="coerce"
    )

    data[iv_column] = (
        data[iv_column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    data[iv_column] = pd.to_numeric(
        data[iv_column], errors="coerce"
    )

    data = data.dropna(
        subset=[date_column, iv_column]
    ).copy()

    data = data[data[iv_column] > 0].copy()

    if data.empty:
        raise ValueError("No valid IV observations were found.")

    # Bloomberg IV is quoted as an annual percentage.
    # Convert values like 30 into 0.30.
    data["annual_iv"] = data[iv_column].apply(
        lambda value: value / 100 if value > 3 else value
    )

    data = data.rename(columns={date_column: "Date"})
    data = data[["Date", "annual_iv"]].copy()

    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)

    data = data.sort_values("Date")
    data = data.drop_duplicates(subset="Date", keep="last")
    data = data.set_index("Date")

    return data


# Download the NVDA price data
def load_prices():
    start_date = (
        pd.Timestamp(backtest_start)
        - pd.DateOffset(years=training_years, months=2)
    ).strftime("%Y-%m-%d")

    end_date = (
        pd.Timestamp(backtest_end) + pd.Timedelta(days=20)
    ).strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
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
    prices = prices[~prices.index.duplicated(keep="last")]

    return prices


# Run one five-day forecast (same logic as the live forecast script)
def run_forecast(prices, iv_data, forecast_index, rng):
    forecast_date = prices.index[forecast_index]
    end_index = forecast_index + forecast_days

    if end_index >= len(prices):
        return None

    actual_end_date = prices.index[end_index]

    start_price = float(prices.iloc[forecast_index])
    actual_price = float(prices.iloc[end_index])

    # Use the previous five years of prices to fit the model
    training_start = forecast_date - pd.DateOffset(years=training_years)

    training_prices = prices.loc[
        (prices.index >= training_start)
        & (prices.index <= forecast_date)
    ]

    # Work with percentage log returns
    returns = (
        np.log(training_prices / training_prices.shift(1))
        .dropna() * 100
    )

    if len(returns) < 250:
        return None

    # Fit GJR-GARCH(1,1) with Student-t errors
    model = arch_model(
        returns,
        mean="Zero",
        p=1,
        o=1,
        q=1,
        dist="t",
        rescale=False
    )

    fitted_model = model.fit(
        disp="off",
        show_warning=False
    )

    if fitted_model.convergence_flag != 0:
        return None

    parameters = fitted_model.params

    omega = float(parameters["omega"])
    alpha = float(parameters["alpha[1]"])
    gamma = float(parameters["gamma[1]"])
    beta = float(parameters["beta[1]"])
    nu = float(parameters["nu"])

    if nu <= 2:
        raise ValueError("Student-t degrees of freedom must exceed 2.")

    last_variance = float(fitted_model.conditional_volatility.iloc[-1] ** 2)
    last_resid = float(fitted_model.resid.iloc[-1])

    # Only use IV data available on or before the forecast date
    available_iv = iv_data.loc[iv_data.index <= forecast_date]

    if available_iv.empty:
        return None

    iv_date = available_iv.index[-1]
    annual_iv = float(available_iv["annual_iv"].iloc[-1])

    # Daily IV variance in percentage-squared units
    iv_daily_variance = (annual_iv * 100) ** 2 / 252

    # Standardised Student-t shocks
    shocks = (
        rng.standard_t(nu, size=(n_simulations, forecast_days))
        / np.sqrt(nu / (nu - 2))
    )

    simulated_prices = np.full(n_simulations, start_price)
    garch_variance = np.full(n_simulations, last_variance)
    previous_resid = np.full(n_simulations, last_resid)

    daily_variance = []

    for day in range(forecast_days):
        # 1. Update GJR-GARCH conditional variance
        indicator = (previous_resid < 0).astype(float)

        garch_variance = (
            omega
            + alpha * previous_resid**2
            + gamma * indicator * previous_resid**2
            + beta * garch_variance
        )

        garch_variance = np.maximum(garch_variance, 1e-12)

        # 2. Blend GARCH variance with constant IV variance
        blended_variance = (
            garch_weight * garch_variance
            + (1 - garch_weight) * iv_daily_variance
        )

        blended_variance = np.maximum(blended_variance, 1e-12)
        daily_variance.append(float(np.mean(blended_variance)))

        # 3. GARCH residual drives the GARCH recursion
        garch_resid = np.sqrt(garch_variance) * shocks[:, day]

        # 4. Blended volatility drives the simulated price changes
        blended_resid = np.sqrt(blended_variance) * shocks[:, day]

        simulated_prices *= np.exp(
            blended_resid / 100 - 0.5 * blended_variance / 10000
        )

        previous_resid = garch_resid

    median_price = float(np.percentile(simulated_prices, 50))
    lower_bound = float(np.percentile(simulated_prices, 5))
    upper_bound = float(np.percentile(simulated_prices, 95))

    # 5-day horizon volatility in %, same measure as the other scripts
    forecast_volatility = float(np.sqrt(np.sum(daily_variance)))

    period_prices = prices.iloc[forecast_index:end_index + 1]
    period_returns = (
        np.log(period_prices / period_prices.shift(1)).dropna() * 100
    )
    realised_volatility = float(np.sqrt(np.sum(period_returns**2)))

    price_error = abs(median_price - actual_price)
    percentage_error = price_error / actual_price * 100

    return {
        "Forecast Date": forecast_date,
        "Actual End Date": actual_end_date,
        "Training Observations": len(returns),
        "IV Observation Date": iv_date,
        "Annualised IV (%)": annual_iv * 100,
        "GARCH Weight (%)": garch_weight * 100,
        "IV Weight (%)": (1 - garch_weight) * 100,
        "Starting Price": start_price,
        "Actual Price": actual_price,
        "Predicted Median Price": median_price,
        "90% Lower Bound": lower_bound,
        "90% Upper Bound": upper_bound,
        "Absolute Price Error": price_error,
        "Percentage Price Error (%)": percentage_error,
        "90% Interval Covered": (
            lower_bound <= actual_price <= upper_bound
        ),
        "90% Interval Width": upper_bound - lower_bound,
        "Forecast Volatility (%)": forecast_volatility,
        "Realised Volatility (%)": realised_volatility,
        "Absolute Volatility Error (%)": abs(
            forecast_volatility - realised_volatility
        ),
        "Degrees of Freedom": nu
    }


# Update the shared summary without removing other models
def update_summary(results):
    model_name = "GJR-GARCH-t-IV"

    summary_row = {
        "model": model_name,
        "forecast_runs": len(results),
        "simulations_per_run": n_simulations,
        "median_absolute_price_error": results[
            "Absolute Price Error"
        ].median(),
        "mean_absolute_price_error": results[
            "Absolute Price Error"
        ].mean(),
        "root_mean_squared_price_error": np.sqrt(
            np.mean(results["Absolute Price Error"] ** 2)
        ),
        "mean_absolute_percentage_error_pct": results[
            "Percentage Price Error (%)"
        ].mean(),
        "90_percent_interval_coverage_pct": (
            results["90% Interval Covered"].mean() * 100
        ),
        "average_interval_width": results[
            "90% Interval Width"
        ].mean(),
        "mean_absolute_volatility_error": results[
            "Absolute Volatility Error (%)"
        ].mean()
    }

    if summary_file.exists():
        summary = pd.read_csv(summary_file)

        if "model" in summary.columns:
            summary = summary[
                summary["model"] != model_name
            ].copy()
        else:
            summary = pd.DataFrame()
    else:
        summary = pd.DataFrame()

    for column in summary_row:
        if column not in summary.columns:
            summary[column] = np.nan

    summary = pd.concat(
        [summary, pd.DataFrame([summary_row])],
        ignore_index=True
    )

    summary.to_csv(summary_file, index=False)



# Plot the results
def create_graphs(results):
    dates = pd.to_datetime(results["Actual End Date"])

    plt.figure(figsize=(12, 6))
    plt.plot(dates, results["Actual Price"], label="Actual Price")
    plt.plot(
        dates, results["Predicted Median Price"],
        label="Predicted Median Price"
    )
    plt.fill_between(
        dates,
        results["90% Lower Bound"].to_numpy(dtype=float),
        results["90% Upper Bound"].to_numpy(dtype=float),
        alpha=0.2,
        label="90% Prediction Interval"
    )
    plt.title("NVDA GJR-GARCH-t-IV Price Backtest")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(price_plot_file, dpi=150)
    plt.close()


# Run the backtest
def main():
    print("Loading Bloomberg IV data...")
    iv_data = load_iv(iv_file)

    print("Loading NVDA prices...")
    prices = load_prices()

    start_date = pd.Timestamp(backtest_start)
    end_date = pd.Timestamp(backtest_end)

    forecast_indices = [
        i for i, date in enumerate(prices.index)
        if start_date <= date < end_date
        and i + forecast_days < len(prices)
        and prices.index[i + forecast_days] <= end_date
    ]

    forecast_indices = forecast_indices[::forecast_step]

    if not forecast_indices:
        raise ValueError("No forecast dates were found.")

    rng = np.random.default_rng(random_seed)
    all_results = []

    for number, index in enumerate(forecast_indices, start=1):
        date = prices.index[index]

        try:
            result = run_forecast(prices, iv_data, index, rng)

            if result is not None:
                all_results.append(result)
                print(
                    f"{number}/{len(forecast_indices)} "
                    f"Completed: {date.date()}"
                )
            else:
                print(f"Skipped: {date.date()}")

        except Exception as error:
            print(f"Failed on {date.date()}: {error}")

    if not all_results:
        raise RuntimeError(
            "No forecasts completed. Check the IV file and its columns."
        )

    results = pd.DataFrame(all_results)

    results.to_csv(results_file, index=False)

    update_summary(results)
    create_graphs(results)

    print("\nBacktest completed.")

if __name__ == "__main__":
    main()