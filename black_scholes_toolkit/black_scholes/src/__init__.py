"""
black_scholes — A professional Python toolkit for options pricing and risk analysis.
====================================================================================

Modules
-------
black_scholes   : Core BS pricing formula, Greeks, implied volatility
greeks_analysis : Sensitivity tables, payoff diagrams, delta-hedge simulation
volatility_surface : IV smile & surface construction
visualisation   : All plotting functions

Quick start
-----------
    from src import black_scholes as bs

    params = bs.OptionParameters(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type='call')
    print(bs.price(params))
    print(bs.all_greeks(params))
"""

from .black_scholes import (
    OptionParameters,
    price,
    delta,
    gamma,
    theta,
    vega,
    rho,
    all_greeks,
    implied_volatility,
    put_call_parity_check,
)

from .greeks_analysis import (
    greeks_vs_spot,
    greeks_vs_time,
    greeks_vs_volatility,
    payoff_diagram,
    delta_hedge_simulation,
    option_summary,
)

from .volatility_surface import (
    iv_smile,
    volatility_surface,
    skew_metrics,
)

from . import visualisation

__all__ = [
    "OptionParameters",
    "price", "delta", "gamma", "theta", "vega", "rho",
    "all_greeks", "implied_volatility", "put_call_parity_check",
    "greeks_vs_spot", "greeks_vs_time", "greeks_vs_volatility",
    "payoff_diagram", "delta_hedge_simulation", "option_summary",
    "iv_smile", "volatility_surface", "skew_metrics",
    "visualisation",
]
