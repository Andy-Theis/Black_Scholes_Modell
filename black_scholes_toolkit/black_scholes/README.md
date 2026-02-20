# Black-Scholes Options Pricing Toolkit

A professional Python implementation of the **Black-Scholes model** for European options pricing and risk analysis — built for use in quantitative finance and banking contexts.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Features

| Module | Description |
|---|---|
| `src/black_scholes.py` | Core BS formula: pricing, all Greeks, implied volatility, put-call parity |
| `src/greeks_analysis.py` | Sensitivity tables, payoff diagrams, delta-hedge simulation |
| `src/volatility_surface.py` | IV smile, full 3-D surface, skew metrics |
| `src/visualisation.py` | Publication-quality plots (6 chart types) |

**Highlights:**
- European Call & Put pricing with continuous dividend yield
- All five Greeks: Δ Delta, Γ Gamma, Θ Theta, ν Vega, ρ Rho
- Implied Volatility via Brent root-finding method
- Delta-hedging P&L simulation (GBM path, discrete rebalancing)
- Parametric volatility smile & 3-D surface (skew + term structure)
- 34 unit tests with pytest
- Interactive CLI and full demo mode

---

## Quickstart

### Installation

```bash
git clone https://github.com/yourusername/black-scholes-toolkit.git
cd black-scholes-toolkit
pip install -r requirements.txt
```

### Quick Example

```python
from src.black_scholes import OptionParameters, price, all_greeks, implied_volatility

# Define a 3-month call option
params = OptionParameters(
    S=100,          # Spot price
    K=105,          # Strike
    T=0.25,         # 3 months
    r=0.05,         # 5% risk-free rate
    sigma=0.20,     # 20% volatility
    option_type="call"
)

# Price it
print(f"Option Price: {price(params):.4f}")   # → 2.8446

# Get all Greeks
greeks = all_greeks(params)
print(greeks)
# {'delta': 0.3785, 'gamma': 0.0351, 'theta': -0.0185, 'vega': 0.1924, 'rho': 0.0873}

# Back out implied volatility from a market price
iv = implied_volatility(market_price=3.50, S=100, K=105, T=0.25, r=0.05, option_type="call")
print(f"Implied Vol: {iv*100:.2f}%")          # → 22.73%
```

### Run the full demo

```bash
python main.py --demo
```

This generates all charts in the `output/` directory:

| File | Content |
|---|---|
| `dashboard.png` | 6-panel summary for one option |
| `greeks_vs_spot.png` | All Greeks as function of spot price |
| `payoff.png` | Payoff & P&L diagram at expiry |
| `theta_decay.png` | Time decay visualisation |
| `iv_smile.png` | Volatility smile |
| `vol_surface.png` | 3-D volatility surface + heatmap |
| `delta_hedge.png` | Delta-hedging simulation |

### Interactive calculator

```bash
python main.py --interactive
```

### Run unit tests

```bash
pytest tests/ -v
```

---

## Sample Outputs

### Dashboard
![dashboard](output/dashboard.png)

### Volatility Surface
![surface](output/vol_surface.png)

### Delta-Hedging Simulation
![hedge](output/delta_hedge.png)

---

## Module Reference

### `OptionParameters`

```python
OptionParameters(
    S: float,                    # Spot price
    K: float,                    # Strike price
    T: float,                    # Time to expiry (years)
    r: float,                    # Risk-free rate (annualised)
    sigma: float,                # Volatility (annualised)
    q: float = 0.0,              # Dividend yield
    option_type: str = "call"    # "call" or "put"
)
```

### Pricing

```python
from src.black_scholes import price
price(params)                    # → float
```

### Greeks

```python
from src.black_scholes import delta, gamma, theta, vega, rho, all_greeks

delta(params)   # Δ — hedge ratio, range [0,1] calls / [-1,0] puts
gamma(params)   # Γ — rate of change of delta
theta(params)   # Θ — time decay per calendar day
vega(params)    # ν — sensitivity to 1% vol change
rho(params)     # ρ — sensitivity to 1% rate change
all_greeks(params)   # → dict with all five
```

### Implied Volatility

```python
from src.black_scholes import implied_volatility

iv = implied_volatility(
    market_price=3.50,
    S=100, K=105, T=0.25, r=0.05,
    option_type="call"
)
```

### Volatility Surface

```python
from src.volatility_surface import iv_smile, volatility_surface

smile = iv_smile(S=100, T=0.25, r=0.05, sigma_atm=0.20, skew=-0.10)
surface = volatility_surface(S=100, r=0.05, sigma_atm=0.20, skew=-0.10)
```

### Delta-Hedge Simulation

```python
from src.greeks_analysis import delta_hedge_simulation

sim = delta_hedge_simulation(params, n_steps=252, seed=42, rebalance_frequency=1)
# Returns DataFrame with daily: S, delta, portfolio_value, hedge_error, ...
```

---

## Mathematical Background

### Black-Scholes Formula

For a European **call** option:

```
C = S·e^(-qT)·N(d₁) - K·e^(-rT)·N(d₂)

d₁ = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
d₂ = d₁ - σ·√T
```

For a European **put**:

```
P = K·e^(-rT)·N(-d₂) - S·e^(-qT)·N(-d₁)
```

Where:
- `N(·)` = cumulative standard normal distribution
- `S` = spot price, `K` = strike, `T` = time (years)
- `r` = risk-free rate, `q` = dividend yield, `σ` = volatility

### Put-Call Parity

```
C - P = S·e^(-qT) - K·e^(-rT)
```

### Key Assumptions

1. Underlying follows Geometric Brownian Motion (log-normal returns)
2. Constant volatility over the option's life
3. No transaction costs, continuous hedging possible
4. European exercise only (no early exercise)
5. Constant risk-free rate

---

## Project Structure

```
black-scholes-toolkit/
│
├── src/
│   ├── __init__.py             # Package exports
│   ├── black_scholes.py        # Core model
│   ├── greeks_analysis.py      # Sensitivity analysis
│   ├── volatility_surface.py   # Vol smile & surface
│   └── visualisation.py        # All plots
│
├── tests/
│   └── test_black_scholes.py   # 34 unit tests
│
├── output/                     # Generated charts
│
├── main.py                     # CLI entry point
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
seaborn>=0.12
pytest>=7.0
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

Andreas Theis — atheis87@googlemail.com — [LinkedIn](www.linkedin.com/in/andreas-theis-261a46384) — [GitHub](https://github.com)
