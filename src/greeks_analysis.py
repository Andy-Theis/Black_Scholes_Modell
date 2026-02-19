"""
greeks_analysis.py
==================
Greeks sensitivity analysis and payoff diagrams.

Covers:
    - Greeks as a function of spot / strike / time / vol
    - P&L simulation (Delta-Hedging P&L)
    - Option payoff at expiry

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from typing import Optional
from .black_scholes import OptionParameters, price, delta, gamma, theta, vega, rho, all_greeks


# ---------------------------------------------------------------------------
# Greeks sensitivity tables
# ---------------------------------------------------------------------------

def greeks_vs_spot(
    params: OptionParameters,
    spot_range: Optional[np.ndarray] = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Compute all Greeks across a range of underlying prices.

    Parameters
    ----------
    params : OptionParameters
        Base parameters. S will be varied.
    spot_range : array-like, optional
        Custom spot price range. Defaults to ±40% around params.S.
    n_points : int
        Number of price points.

    Returns
    -------
    pd.DataFrame
    """
    if spot_range is None:
        spot_range = np.linspace(params.S * 0.60, params.S * 1.40, n_points)

    rows = []
    for s in spot_range:
        p = OptionParameters(
            S=s, K=params.K, T=params.T, r=params.r,
            sigma=params.sigma, q=params.q, option_type=params.option_type
        )
        greeks = all_greeks(p)
        rows.append({
            "spot": s,
            "option_price": price(p),
            "intrinsic_value": max(s - params.K, 0) if params.option_type == "call" else max(params.K - s, 0),
            **greeks,
        })

    df = pd.DataFrame(rows)
    df["time_value"] = df["option_price"] - df["intrinsic_value"]
    return df


def greeks_vs_time(
    params: OptionParameters,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Compute all Greeks across remaining time to expiry (from params.T down to 0).

    Useful for visualising time decay (Theta).
    """
    times = np.linspace(params.T, 0.001, n_points)

    rows = []
    for t in times:
        p = OptionParameters(
            S=params.S, K=params.K, T=t, r=params.r,
            sigma=params.sigma, q=params.q, option_type=params.option_type
        )
        greeks = all_greeks(p)
        rows.append({
            "days_to_expiry": round(t * 365, 1),
            "T": t,
            "option_price": price(p),
            **greeks,
        })

    return pd.DataFrame(rows)


def greeks_vs_volatility(
    params: OptionParameters,
    vol_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute all Greeks across a range of volatility values.
    """
    if vol_range is None:
        vol_range = np.linspace(0.05, 0.80, 76)

    rows = []
    for sigma in vol_range:
        p = OptionParameters(
            S=params.S, K=params.K, T=params.T, r=params.r,
            sigma=sigma, q=params.q, option_type=params.option_type
        )
        greeks = all_greeks(p)
        rows.append({
            "volatility_pct": sigma * 100,
            "option_price": price(p),
            **greeks,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Payoff diagram
# ---------------------------------------------------------------------------

def payoff_diagram(
    params: OptionParameters,
    spot_range: Optional[np.ndarray] = None,
    cost: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute option payoff and P&L at expiry.

    Parameters
    ----------
    params : OptionParameters
    spot_range : array-like, optional
        Spot prices at expiry to evaluate. Defaults to ±40% around S.
    cost : float, optional
        Premium paid (for P&L calculation). If None, uses theoretical BS price.

    Returns
    -------
    pd.DataFrame with columns: spot_at_expiry, payoff, pnl
    """
    if spot_range is None:
        spot_range = np.linspace(params.S * 0.60, params.S * 1.40, 200)

    if cost is None:
        cost = price(params)

    if params.option_type == "call":
        payoffs = np.maximum(spot_range - params.K, 0)
    else:
        payoffs = np.maximum(params.K - spot_range, 0)

    return pd.DataFrame({
        "spot_at_expiry": spot_range,
        "payoff": payoffs,
        "pnl": payoffs - cost,
        "breakeven": params.K + cost if params.option_type == "call" else params.K - cost,
    })


# ---------------------------------------------------------------------------
# Delta-Hedging P&L simulation
# ---------------------------------------------------------------------------

def delta_hedge_simulation(
    params: OptionParameters,
    n_steps: int = 252,
    seed: int = 42,
    rebalance_frequency: int = 1,
) -> pd.DataFrame:
    """
    Simulate a discrete Delta-Hedging strategy over the option's life.

    The bank has sold the option and hedges by holding Delta shares.
    We simulate a GBM path for the underlying and track:
        - Portfolio value
        - Hedging cost
        - P&L vs theoretical BS price

    Parameters
    ----------
    params : OptionParameters
        Initial parameters. T is the simulation horizon.
    n_steps : int
        Number of daily steps (default 252 trading days).
    seed : int
        Random seed for reproducibility.
    rebalance_frequency : int
        How often (in steps) to rebalance. 1 = daily, 5 = weekly.

    Returns
    -------
    pd.DataFrame
        Day-by-day simulation results.
    """
    rng = np.random.default_rng(seed)
    dt = params.T / n_steps
    S0 = params.S

    # Initial option price (received as premium)
    initial_premium = price(params)

    # Simulate GBM path
    z = rng.standard_normal(n_steps)
    log_returns = (params.r - 0.5 * params.sigma**2) * dt + params.sigma * np.sqrt(dt) * z
    S_path = S0 * np.exp(np.cumsum(np.concatenate([[0], log_returns])))

    rows = []
    cash = initial_premium  # cash account starts with received premium
    shares_held = 0.0
    cumulative_hedge_cost = 0.0

    for i in range(n_steps + 1):
        t_remaining = params.T - i * dt
        t_remaining = max(t_remaining, 1e-6)
        S_t = S_path[i]

        p_t = OptionParameters(
            S=S_t, K=params.K, T=t_remaining,
            r=params.r, sigma=params.sigma, q=params.q,
            option_type=params.option_type
        )
        current_option_price = price(p_t)
        current_delta = delta(p_t)
        current_gamma = gamma(p_t)

        # Rebalance hedge
        if i % rebalance_frequency == 0 or i == 0:
            delta_change = current_delta - shares_held
            hedge_trade_cost = delta_change * S_t
            cash -= hedge_trade_cost  # buy/sell shares
            cash *= np.exp(params.r * dt * rebalance_frequency)  # accrue interest
            cumulative_hedge_cost += abs(hedge_trade_cost)
            shares_held = current_delta

        portfolio_value = shares_held * S_t + cash  # long shares + cash
        hedge_error = portfolio_value - current_option_price  # should be ~0

        rows.append({
            "day": i,
            "S": round(S_t, 4),
            "option_price": round(current_option_price, 4),
            "delta": round(current_delta, 4),
            "gamma": round(current_gamma, 6),
            "shares_held": round(shares_held, 4),
            "cash": round(cash, 4),
            "portfolio_value": round(portfolio_value, 4),
            "hedge_error": round(hedge_error, 4),
            "cumulative_hedge_cost": round(cumulative_hedge_cost, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics for a parameter set
# ---------------------------------------------------------------------------

def option_summary(params: OptionParameters) -> dict:
    """
    Full summary of an option: price, Greeks, moneyness, breakeven.

    Returns
    -------
    dict
    """
    opt_price = price(params)
    greeks = all_greeks(params)
    moneyness = params.S / params.K

    if params.option_type == "call":
        breakeven = params.K + opt_price
        intrinsic = max(params.S - params.K, 0)
    else:
        breakeven = params.K - opt_price
        intrinsic = max(params.K - params.S, 0)

    return {
        "option_type": params.option_type.upper(),
        "spot_S": params.S,
        "strike_K": params.K,
        "maturity_T_years": params.T,
        "maturity_T_days": round(params.T * 365),
        "risk_free_rate_pct": params.r * 100,
        "volatility_pct": params.sigma * 100,
        "dividend_yield_pct": params.q * 100,
        "option_price": round(opt_price, 4),
        "intrinsic_value": round(intrinsic, 4),
        "time_value": round(opt_price - intrinsic, 4),
        "moneyness": round(moneyness, 4),
        "moneyness_pct": round((moneyness - 1) * 100, 2),
        "breakeven_at_expiry": round(breakeven, 4),
        **{k: round(v, 6) for k, v in greeks.items()},
    }
