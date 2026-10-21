import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

tkr = "NVDA"
print(f"Fetching {tkr} data...")

# ---- Forecast settings ----
days = 5
w_garch = 0.5
n_sim = 100000
n_plot = n_sim  # Plot all 100,000 paths

# ---- Historical price data ----
df = yf.download(
    tkr,
    period="5y",
    progress=False
)["Close"]

if isinstance(df, pd.DataFrame):
    df = df.iloc[:, 0]

df = df.dropna()

if len(df) < 100:
    raise ValueError("Insufficient historical price data.")

curr_px = float(df.iloc[-1])

# Last date with available historical price data
last_market_date = pd.Timestamp(df.index[-1]).normalize()

# Generate the next five weekdays, excluding weekends
forecast_dates = pd.bdate_range(
    start=last_market_date + pd.Timedelta(days=1),
    periods=days
)

print(f"Last available market date: {last_market_date:%Y-%m-%d}")
print(
    f"Forecast dates: {forecast_dates[0]:%Y-%m-%d} to "
    f"{forecast_dates[-1]:%Y-%m-%d}"
)

# ---- Historical returns ----
rets = np.log(df / df.shift(1)).dropna() * 100

# ---- Fit GJR-GARCH(1,1) with Student-t errors ----
am = arch_model(
    rets,
    mean="Zero",
    p=1,
    o=1,
    q=1,
    dist="t",
    rescale=False
)

res = am.fit(disp="off")

assert res.convergence_flag == 0, "GARCH did not converge."

# ---- Options data ----
ticker = yf.Ticker(tkr)

start = (
    last_market_date + pd.Timedelta(days=1)
).date()

exps = [
    d for d in ticker.options
    if pd.to_datetime(d) > last_market_date
]

if not exps:
    raise ValueError("No future option expirations found.")

# Select expiry closest to the five-trading-day horizon
exp_days = []

for d in exps:
    end = (pd.to_datetime(d) + pd.Timedelta(days=1)).date()
    exp_days.append(np.busday_count(start, end))

target_idx = int(
    np.argmin([abs(x - days) for x in exp_days])
)

target_exp = exps[target_idx]
iv_days = exp_days[target_idx]

opt = ticker.option_chain(target_exp)

# ---- Estimate IV from OTM options ----
max_moneyness = 0.10
min_oi = 10
max_rel_spread = 0.5
kernel_bw = 0.03

calls = opt.calls.copy()
puts = opt.puts.copy()

calls = calls[calls["strike"] >= curr_px]
puts = puts[puts["strike"] < curr_px]

chain = pd.concat([calls, puts], ignore_index=True)

chain["mid"] = (chain["bid"] + chain["ask"]) / 2
chain["rel_spread"] = (
    (chain["ask"] - chain["bid"]) / chain["mid"]
)
chain["log_m"] = np.log(chain["strike"] / curr_px)

good = chain[
    (chain["impliedVolatility"] > 0.05)
    & (chain["bid"] > 0)
    & (chain["ask"] > chain["bid"])
    & (chain["rel_spread"] < max_rel_spread)
    & (chain["openInterest"].fillna(0) >= min_oi)
    & (chain["log_m"].abs() <= max_moneyness)
].copy()

n_iv_contracts = len(good)

if n_iv_contracts == 0:
    print("\nNo usable options contracts; using GARCH only.")
    atm_iv = np.nan
    iv_daily_var = 0.0
    iv_vol = np.nan
else:
    weights = np.exp(
        -0.5 * (good["log_m"] / kernel_bw) ** 2
    )

    atm_iv = float(np.sqrt(
        np.average(
            good["impliedVolatility"] ** 2,
            weights=weights
        )
    ))

    # Daily variance in percentage-squared units
    iv_daily_var = (atm_iv * 100) ** 2 / 252
    iv_vol = atm_iv * np.sqrt(iv_days / 252)

    print(f"\nIV contracts used: {n_iv_contracts}")
    print(f"Weighted annualised IV: {atm_iv:.2%}")

    if n_iv_contracts < 6:
        print("Warning: IV estimate may be noisy.")

print(f"Target expiry: {target_exp} ({iv_days} trading days)")
print(f"Starting price: ${curr_px:.2f}")

if abs(iv_days - days) > 1:
    print("Warning: option expiry differs from forecast horizon.")

# ---- GJR-GARCH parameters ----
omega = float(res.params["omega"])
alpha = float(res.params["alpha[1]"])
gamma = float(res.params["gamma[1]"])
beta = float(res.params["beta[1]"])
nu = float(res.params["nu"])

assert nu > 2, "Student-t degrees of freedom must exceed 2."

last_var = float(res.conditional_volatility.iloc[-1] ** 2)
last_resid = float(res.resid.iloc[-1])

if not 0 <= w_garch <= 1:
    raise ValueError("w_garch must be between 0 and 1.")

# ---- Monte Carlo simulation ----
np.random.seed(42)

shocks = (
    np.random.standard_t(nu, size=(n_sim, days))
    / np.sqrt(nu / (nu - 2))
)

prices = np.zeros((n_sim, days + 1))
prices[:, 0] = curr_px

var = np.full(n_sim, last_var)
prev_resid = np.full(n_sim, last_resid)

daily_vols = []

for j in range(days):

    # 1. Update GJR-GARCH conditional variance
    indicator = (prev_resid < 0).astype(float)

    var = (
        omega
        + alpha * prev_resid**2
        + gamma * indicator * prev_resid**2
        + beta * var
    )

    var = np.maximum(var, 1e-12)

    # 2. Blend GARCH variance with constant IV variance
    if iv_daily_var > 0:
        sim_var = (
            w_garch * var
            + (1 - w_garch) * iv_daily_var
        )
    else:
        sim_var = var

    sim_var = np.maximum(sim_var, 1e-12)
    daily_vols.append(float(np.sqrt(np.mean(sim_var))))

    # 3. GARCH residual updates the GARCH recursion
    garch_resid = np.sqrt(var) * shocks[:, j]

    # 4. Blended volatility drives simulated price changes
    # No directional drift or momentum adjustment
    sim_resid = np.sqrt(sim_var) * shocks[:, j]

    prices[:, j + 1] = prices[:, j] * np.exp(
        sim_resid / 100 - 0.5 * sim_var / 10000
    )
    prev_resid = garch_resid

# ---- Results ----
print("\n--- Monte Carlo Forecast ---")
print(f"Simulations: {n_sim:,}")
print(f"Paths plotted: {n_plot:,}")
print(f"Forecast horizon: {days} trading days")
print("Directional drift: None (zero-mean log-return shocks)")
print(f"GARCH variance weight: {w_garch:.2f}")

for j, forecast_date in enumerate(forecast_dates, start=1):
    day_prices = prices[:, j]

    print(f"\n{forecast_date:%A, %d %B %Y} Forecast:")
    print(f"Median price:      ${np.percentile(day_prices, 50):.2f}")
    print(f"Mean price:        ${np.mean(day_prices):.2f}")
    print(f"5th percentile:    ${np.percentile(day_prices, 5):.2f}")
    print(f"95th percentile:   ${np.percentile(day_prices, 95):.2f}")

# ---- Volatility check ----
final_prices = prices[:, -1]

sim_std = np.std(
    np.log(final_prices / curr_px),
    ddof=1
)

print("\n--- Volatility Check ---")
print(f"Simulated {days}-day log-return volatility: {sim_std:.2%}")

if not np.isnan(iv_vol):
    print(f"Implied {days}-day volatility: {iv_vol:.2%}")

print("\nMean blended daily volatility:")
for forecast_date, vol in zip(forecast_dates, daily_vols):
    print(f"{forecast_date:%Y-%m-%d}: {vol:.4f}%")

# ---- Plot all 100,000 paths ----
plot_dates = [last_market_date] + list(forecast_dates)

plt.figure(figsize=(12, 7))

for i in range(n_plot):
    plt.plot(
        range(days + 1),
        prices[i],
        alpha=0.5,
        linewidth=1
    )

plt.title(
    f"Monte Carlo Forecast: GJR-GARCH(1,1) + IV ({tkr})"
)
plt.xlabel("Forecast date")
plt.ylabel("Price ($)")
plt.xticks(
    range(days + 1),
    [d.strftime("%d %b") for d in plot_dates],
    rotation=30
)
plt.grid(False)
plt.tight_layout()
plt.show()