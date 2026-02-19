"""
black_scholes.py
================
Core Black-Scholes pricing model for European options.

Implements:
    - Call and Put pricing
    - All Greeks (Delta, Gamma, Theta, Vega, Rho)
    - Implied Volatility via Newton-Raphson
    - Put-Call Parity validation

Author: [Your Name]
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class OptionParameters:
    """
    Container for all Black-Scholes input parameters.

    Attributes
    ----------
    S : float
        Current underlying price (e.g. stock price in €/$ per share).
    K : float
        Strike price.
    T : float
        Time to expiration in years (e.g. 3 months = 0.25).
    r : float
        Risk-free interest rate (annualised, continuous compounding), e.g. 0.05 = 5%.
    sigma : float
        Annualised volatility of the underlying, e.g. 0.20 = 20%.
    q : float, optional
        Continuous dividend yield (default 0.0).
    option_type : str
        'call' or 'put'.
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: Literal["call", "put"] = "call"

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"Underlying price S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike K must be positive, got {self.K}")
        if self.T < 0:
            raise ValueError(f"Time to expiry T must be non-negative, got {self.T}")
        if self.sigma < 0:
            raise ValueError(f"Volatility sigma must be non-negative, got {self.sigma}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{self.option_type}'")


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """
    Compute d1 and d2 from the Black-Scholes formula.

    Returns
    -------
    d1, d2 : tuple of float
        The two standard normal arguments used throughout the BS formula.
    """
    if T == 0 or sigma == 0:
        # At expiry or zero vol: option is worth its intrinsic value
        return np.inf, np.inf

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(params: OptionParameters) -> float:
    """
    Calculate the Black-Scholes price of a European option.

    Parameters
    ----------
    params : OptionParameters

    Returns
    -------
    float
        Theoretical option price.

    Examples
    --------
    >>> p = OptionParameters(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type='call')
    >>> round(price(p), 4)
    2.8446
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q

    if T == 0:
        if params.option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discount = np.exp(-r * T)
    forward_discount = np.exp(-q * T)

    if params.option_type == "call":
        return S * forward_discount * norm.cdf(d1) - K * discount * norm.cdf(d2)
    else:
        return K * discount * norm.cdf(-d2) - S * forward_discount * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(params: OptionParameters) -> float:
    """
    Delta: sensitivity of option price to a €1 change in the underlying.

    Range: [0, 1] for calls, [-1, 0] for puts.
    Interpretation: How many units of the underlying are needed to hedge one option.
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    if T == 0:
        if params.option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if params.option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(params: OptionParameters) -> float:
    """
    Gamma: rate of change of Delta with respect to the underlying price.

    Same for calls and puts.
    High gamma = Delta changes rapidly = more frequent re-hedging required.
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    if T == 0 or sigma == 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(params: OptionParameters) -> float:
    """
    Theta: time decay – change in option value per calendar day.

    Returns value per day (divided by 365).
    Typically negative: options lose value as time passes.
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    if T == 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if params.option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)

    return (term1 + term2 + term3) / 365  # per calendar day


def vega(params: OptionParameters) -> float:
    """
    Vega: sensitivity to a 1% change in implied volatility.

    Same for calls and puts.
    Returned value is per 1% change (divided by 100).
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    if T == 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # per 1%


def rho(params: OptionParameters) -> float:
    """
    Rho: sensitivity to a 1% change in the risk-free interest rate.

    Returned value is per 1% change (divided by 100).
    Calls have positive rho (benefit from higher rates), puts negative.
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    if T == 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if params.option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def all_greeks(params: OptionParameters) -> dict:
    """
    Compute all Greeks at once and return as a dictionary.

    Returns
    -------
    dict with keys: delta, gamma, theta, vega, rho
    """
    return {
        "delta": delta(params),
        "gamma": gamma(params),
        "theta": theta(params),
        "vega":  vega(params),
        "rho":   rho(params),
    }


# ---------------------------------------------------------------------------
# Implied Volatility
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    tol: float = 1e-6,
) -> float:
    """
    Calculate implied volatility using the Brent root-finding method.

    Solves: BS_price(sigma) - market_price = 0

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    float
        Implied volatility (annualised), or np.nan if no solution found.

    Raises
    ------
    ValueError
        If market_price violates no-arbitrage bounds.
    """
    # Basic arbitrage bound check
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == "call" else max(K * np.exp(-r * T) - S, 0)
    if market_price < intrinsic - tol:
        raise ValueError(
            f"Market price {market_price:.4f} is below intrinsic value {intrinsic:.4f}. "
            "Arbitrage violation."
        )

    def objective(sigma):
        p = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
        return price(p) - market_price

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=1000)
        return iv
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Put-Call Parity
# ---------------------------------------------------------------------------

def put_call_parity_check(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 0.01,
) -> dict:
    """
    Verify Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)

    Returns
    -------
    dict with 'lhs', 'rhs', 'difference', 'holds' (bool)
    """
    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    diff = abs(lhs - rhs)
    return {
        "lhs (C - P)": round(lhs, 6),
        "rhs (S·e^-qT - K·e^-rT)": round(rhs, 6),
        "difference": round(diff, 6),
        "parity_holds": diff < tol,
    }
