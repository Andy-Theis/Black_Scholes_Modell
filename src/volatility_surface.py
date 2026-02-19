"""
volatility_surface.py
=====================
Tools for computing and visualising the implied volatility surface.

Covers:
    - IV smile for a single expiry
    - Full 3-D surface (Strike × Maturity × IV)
    - Volatility skew metrics

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from .black_scholes import implied_volatility, price, OptionParameters


# ---------------------------------------------------------------------------
# IV Smile (single maturity)
# ---------------------------------------------------------------------------

def iv_smile(
    S: float,
    T: float,
    r: float,
    sigma_atm: float,
    strikes: Optional[np.ndarray] = None,
    skew: float = -0.10,
    smile_convexity: float = 0.05,
    q: float = 0.0,
    option_type: str = "call",
) -> pd.DataFrame:
    """
    Simulate an implied volatility smile for a given maturity.

    Uses a simple parametric model: IV(K) = sigma_ATM + skew*(K-S)/S + curvature*((K-S)/S)^2

    This is a stylised model for illustration.  In production you would
    back out IVs directly from observed market prices.

    Parameters
    ----------
    S : float
        Spot price.
    T : float
        Maturity in years.
    r : float
        Risk-free rate.
    sigma_atm : float
        At-the-money implied volatility.
    strikes : array-like, optional
        Custom strike grid.  Defaults to ±30% moneyness around S.
    skew : float
        Slope of the smile (negative = put skew, typical for equity indices).
    smile_convexity : float
        Curvature / convexity parameter.
    q : float
        Dividend yield.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    pd.DataFrame
        Columns: strike, moneyness, iv, option_price
    """
    if strikes is None:
        strikes = np.linspace(S * 0.70, S * 1.30, 61)

    moneyness = (strikes - S) / S
    iv_surface = sigma_atm + skew * moneyness + smile_convexity * moneyness**2
    iv_surface = np.clip(iv_surface, 0.01, 5.0)

    prices = []
    for K, iv in zip(strikes, iv_surface):
        p = OptionParameters(S=S, K=K, T=T, r=r, sigma=iv, q=q, option_type=option_type)
        prices.append(price(p))

    return pd.DataFrame({
        "strike": strikes,
        "moneyness_pct": moneyness * 100,
        "implied_vol_pct": iv_surface * 100,
        "option_price": prices,
    })


# ---------------------------------------------------------------------------
# Full Volatility Surface (Strikes × Maturities)
# ---------------------------------------------------------------------------

def volatility_surface(
    S: float,
    r: float,
    sigma_atm: float,
    strikes: Optional[np.ndarray] = None,
    maturities: Optional[np.ndarray] = None,
    skew: float = -0.10,
    smile_convexity: float = 0.05,
    term_structure_slope: float = 0.02,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Build a full implied volatility surface across strikes and maturities.

    The parametric model includes:
        - Moneyness skew
        - Smile convexity
        - Term structure (longer maturities typically have higher ATM vol)

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: maturity, strike, moneyness_pct, implied_vol_pct
    """
    if strikes is None:
        strikes = np.linspace(S * 0.70, S * 1.30, 31)
    if maturities is None:
        maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])

    rows = []
    for T in maturities:
        # Term structure: ATM vol increases slightly with maturity
        sigma_atm_t = sigma_atm + term_structure_slope * np.sqrt(T)

        for K in strikes:
            moneyness = (K - S) / S
            iv = sigma_atm_t + skew * moneyness + smile_convexity * moneyness**2
            iv = max(iv, 0.01)
            rows.append({
                "maturity_years": round(T, 4),
                "maturity_label": _maturity_label(T),
                "strike": K,
                "moneyness_pct": round(moneyness * 100, 2),
                "implied_vol_pct": round(iv * 100, 4),
            })

    return pd.DataFrame(rows)


def _maturity_label(T: float) -> str:
    """Convert maturity in years to a readable label."""
    months = round(T * 12)
    if months < 12:
        return f"{months}M"
    elif months == 12:
        return "1Y"
    else:
        years = months / 12
        return f"{years:.1f}Y"


# ---------------------------------------------------------------------------
# Skew Metrics
# ---------------------------------------------------------------------------

def skew_metrics(smile_df: pd.DataFrame, S: float) -> dict:
    """
    Compute common vol skew metrics from a smile DataFrame.

    Parameters
    ----------
    smile_df : pd.DataFrame
        Output of iv_smile().
    S : float
        Spot price.

    Returns
    -------
    dict with skew metrics
    """
    df = smile_df.copy()
    df["abs_moneyness"] = abs(df["moneyness_pct"])

    # ATM vol: closest strike to spot
    atm_row = df.iloc[(df["moneyness_pct"].abs()).argmin()]
    atm_iv = atm_row["implied_vol_pct"]

    # 25-delta skew: IV at 90% moneyness minus 110% moneyness (approximate)
    otm_put = df[df["moneyness_pct"].between(-15, -5)]["implied_vol_pct"].mean()
    otm_call = df[df["moneyness_pct"].between(5, 15)]["implied_vol_pct"].mean()

    return {
        "atm_iv_pct": round(atm_iv, 4),
        "otm_put_iv_avg_pct": round(otm_put, 4),
        "otm_call_iv_avg_pct": round(otm_call, 4),
        "put_call_skew": round(otm_put - otm_call, 4),
        "min_iv_pct": round(df["implied_vol_pct"].min(), 4),
        "max_iv_pct": round(df["implied_vol_pct"].max(), 4),
    }
