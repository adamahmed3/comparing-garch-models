
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

tkr = "NVDA"
print(f"Fetching {tkr} data...")

# ---- Forecast settings ----
days = 5
n_sim = 100000
n_plot = n_sim

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
last_market_date = pd.Timestamp(df.index[-1]).normalize()

# ---- Forecast dates ----
forecast_dates = pd.bdate_range(
    start=last_market_date + pd.Timedelta(days=1),
    periods=days
)

print(f"Last available market date: {last_market_date:%Y-%m-%d}")
print(
    f"Forecast dates: {forecast_dates[0]:%Y-%m-%d} "
    f"to {forecast_dates[-1]:%Y-%m-%d}"
)

# ---- Historical log returns ----
rets = np.log(df / df.shift(1)).dropna() * 100

# ---- Fit GARCH(1,1) ----
am = arch_model(
    rets,
    mean="Zero",
    vol="GARCH",
    p=1,
    o=0,
    q=1,
    dist="normal",
    rescale=False
)

res = am.fit(disp="off")

if res.convergence_flag != 0:
    raise RuntimeError("GARCH did not converge.")

# ---- Model parameters ----
omega = float(res.params["omega"])
alpha = float(res.params["alpha[1]"])
beta = float(res.params["beta[1]"])

last_var = float(res.conditional_volatility.iloc[-1] ** 2)
last_resid = float(res.resid.iloc[-1])
curr_vol = float(res.conditional_volatility.iloc[-1])

print("\n--- Model 1: GARCH(1,1) ---")
print(f"Starting price: ${curr_px:.2f}")
print(f"Historical observations: {len(rets):,}")
print(f"Omega: {omega:.6f}")
print(f"Alpha: {alpha:.6f}")
print(f"Beta: {beta:.6f}")
print(f"Latest fitted daily volatility: {curr_vol:.4f}%")

# ---- Monte Carlo simulation ----
np.random.seed(42)

shocks = np.random.standard_normal(
    size=(n_sim, days)
)

prices = np.zeros((n_sim, days + 1))
prices[:, 0] = curr_px

var = np.full(n_sim, last_var)
prev_resid = np.full(n_sim, last_resid)

daily_vols = []

for j in range(days):

    # Update conditional variance
    var = (
        omega
        + alpha * prev_resid**2
        + beta * var
    )

    var = np.maximum(var, 1e-12)

    daily_vols.append(float(np.sqrt(np.mean(var))))

    # Generate simulated returns
    sim_resid = np.sqrt(var) * shocks[:, j]

    prices[:, j + 1] = prices[:, j] * np.exp(
        sim_resid / 100
    )

    # Update residuals
    prev_resid = sim_resid

# ---- Forecast results ----
print("\n--- Monte Carlo Forecast ---")
print(f"Simulations: {n_sim:,}")
print(f"Paths plotted: {n_plot:,}")
print(f"Forecast horizon: {days} trading days")

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
print(f"Simulated {days}-day log-return volatility: {sim_std:.4f}%")

print("\nMean forecast daily volatility:")
for forecast_date, vol in zip(forecast_dates, daily_vols):
    print(f"{forecast_date:%Y-%m-%d}: {vol:.4f}%")

# ---- Plot Monte Carlo paths ----
plot_dates = [last_market_date] + list(forecast_dates)

plt.figure(figsize=(12, 7))

for i in range(n_plot):
    plt.plot(
        range(days + 1),
        prices[i],
        alpha=0.5,
        linewidth=1
    )

plt.title(f"Monte Carlo Forecast: GARCH(1,1) ({tkr})")
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
