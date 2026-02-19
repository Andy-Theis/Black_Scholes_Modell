"""
test_black_scholes.py
=====================
Unit tests for the Black-Scholes toolkit.

Run with:
    pytest tests/ -v

Author: [Your Name]
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.black_scholes import (
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
from src.greeks_analysis import (
    greeks_vs_spot,
    greeks_vs_time,
    payoff_diagram,
    delta_hedge_simulation,
    option_summary,
)
from src.volatility_surface import iv_smile, volatility_surface, skew_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atm_call():
    """At-the-money call option."""
    return OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")


@pytest.fixture
def atm_put():
    """At-the-money put option."""
    return OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")


@pytest.fixture
def itm_call():
    """In-the-money call."""
    return OptionParameters(S=110, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")


@pytest.fixture
def otm_call():
    """Out-of-the-money call."""
    return OptionParameters(S=90, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------

class TestPricing:
    def test_call_price_positive(self, atm_call):
        assert price(atm_call) > 0

    def test_put_price_positive(self, atm_put):
        assert price(atm_put) > 0

    def test_call_above_intrinsic(self, itm_call):
        opt_price = price(itm_call)
        intrinsic = max(itm_call.S - itm_call.K, 0)
        assert opt_price >= intrinsic - 1e-8

    def test_otm_call_below_atm_call(self, atm_call, otm_call):
        assert price(otm_call) < price(atm_call)

    def test_known_bs_call_price(self):
        """Test against a textbook-known Black-Scholes value."""
        # Hull, Options Futures & Other Derivatives, Example 15.6
        # S=42, K=40, T=0.5, r=0.10, sigma=0.20 → C ≈ 4.76
        p = OptionParameters(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
        assert abs(price(p) - 4.76) < 0.01

    def test_call_at_expiry_itm(self):
        p = OptionParameters(S=110, K=100, T=0.0, r=0.05, sigma=0.20, option_type="call")
        assert abs(price(p) - 10.0) < 1e-8

    def test_call_at_expiry_otm(self):
        p = OptionParameters(S=90, K=100, T=0.0, r=0.05, sigma=0.20, option_type="call")
        assert abs(price(p) - 0.0) < 1e-8

    def test_put_at_expiry_itm(self):
        p = OptionParameters(S=90, K=100, T=0.0, r=0.05, sigma=0.20, option_type="put")
        assert abs(price(p) - 10.0) < 1e-8

    def test_put_call_parity(self, atm_call, atm_put):
        """C - P = S - K*e^(-rT) (no dividends)."""
        C = price(atm_call)
        P = price(atm_put)
        S, K, r, T = atm_call.S, atm_call.K, atm_call.r, atm_call.T
        lhs = C - P
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6

    def test_higher_vol_higher_price(self, atm_call):
        """Increasing volatility should increase option price (positive Vega)."""
        p_low = OptionParameters(S=100, K=100, T=1, r=0.05, sigma=0.10, option_type="call")
        p_high = OptionParameters(S=100, K=100, T=1, r=0.05, sigma=0.40, option_type="call")
        assert price(p_high) > price(p_low)

    def test_longer_maturity_higher_price(self):
        """Longer maturity → higher option price."""
        p_short = OptionParameters(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        p_long = OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
        assert price(p_long) > price(p_short)


# ---------------------------------------------------------------------------
# Greeks tests
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_call_delta_range(self, atm_call):
        d = delta(atm_call)
        assert 0 <= d <= 1

    def test_put_delta_range(self, atm_put):
        d = delta(atm_put)
        assert -1 <= d <= 0

    def test_gamma_positive(self, atm_call):
        assert gamma(atm_call) > 0

    def test_gamma_same_call_put(self, atm_call, atm_put):
        """Gamma is identical for calls and puts with same parameters."""
        assert abs(gamma(atm_call) - gamma(atm_put)) < 1e-10

    def test_theta_negative_call(self, atm_call):
        """Theta should be negative for long options (time decay)."""
        assert theta(atm_call) < 0

    def test_vega_positive(self, atm_call):
        """Vega should be positive (higher vol → higher price)."""
        assert vega(atm_call) > 0

    def test_vega_same_call_put(self, atm_call, atm_put):
        """Vega is identical for calls and puts with same parameters."""
        assert abs(vega(atm_call) - vega(atm_put)) < 1e-10

    def test_rho_positive_call(self, atm_call):
        """Call rho positive: higher rates → higher call price."""
        assert rho(atm_call) > 0

    def test_rho_negative_put(self, atm_put):
        """Put rho negative: higher rates → lower put price."""
        assert rho(atm_put) < 0

    def test_numerical_delta(self, atm_call):
        """Finite difference check for Delta."""
        h = 0.01
        p_up = OptionParameters(S=atm_call.S + h, K=atm_call.K, T=atm_call.T,
                                  r=atm_call.r, sigma=atm_call.sigma, option_type="call")
        p_dn = OptionParameters(S=atm_call.S - h, K=atm_call.K, T=atm_call.T,
                                  r=atm_call.r, sigma=atm_call.sigma, option_type="call")
        numerical = (price(p_up) - price(p_dn)) / (2 * h)
        assert abs(delta(atm_call) - numerical) < 1e-4

    def test_numerical_gamma(self, atm_call):
        """Finite difference check for Gamma."""
        h = 0.01
        p_up = OptionParameters(S=atm_call.S + h, K=atm_call.K, T=atm_call.T,
                                  r=atm_call.r, sigma=atm_call.sigma, option_type="call")
        p_dn = OptionParameters(S=atm_call.S - h, K=atm_call.K, T=atm_call.T,
                                  r=atm_call.r, sigma=atm_call.sigma, option_type="call")
        numerical = (price(p_up) - 2 * price(atm_call) + price(p_dn)) / h**2
        assert abs(gamma(atm_call) - numerical) < 1e-3

    def test_all_greeks_keys(self, atm_call):
        g = all_greeks(atm_call)
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}


# ---------------------------------------------------------------------------
# Implied Volatility tests
# ---------------------------------------------------------------------------

class TestImpliedVolatility:
    def test_iv_recovers_sigma(self, atm_call):
        """BS price → IV → should recover original sigma."""
        market_price = price(atm_call)
        iv = implied_volatility(
            market_price=market_price,
            S=atm_call.S, K=atm_call.K, T=atm_call.T,
            r=atm_call.r, option_type=atm_call.option_type
        )
        assert abs(iv - atm_call.sigma) < 1e-5

    def test_iv_recovers_sigma_put(self, atm_put):
        market_price = price(atm_put)
        iv = implied_volatility(
            market_price=market_price,
            S=atm_put.S, K=atm_put.K, T=atm_put.T,
            r=atm_put.r, option_type=atm_put.option_type
        )
        assert abs(iv - atm_put.sigma) < 1e-5

    def test_iv_itm_call(self, itm_call):
        market_price = price(itm_call)
        iv = implied_volatility(
            market_price=market_price,
            S=itm_call.S, K=itm_call.K, T=itm_call.T,
            r=itm_call.r, option_type=itm_call.option_type
        )
        assert abs(iv - itm_call.sigma) < 1e-5

    def test_iv_arbitrage_violation(self):
        """Should raise ValueError for price below intrinsic."""
        with pytest.raises(ValueError):
            implied_volatility(
                market_price=0.001,  # essentially zero for deep ITM
                S=120, K=100, T=1.0, r=0.05, option_type="call"
            )


# ---------------------------------------------------------------------------
# Put-Call Parity tests
# ---------------------------------------------------------------------------

class TestPutCallParity:
    def test_parity_holds_bs(self, atm_call, atm_put):
        call_p = price(atm_call)
        put_p = price(atm_put)
        result = put_call_parity_check(call_p, put_p, atm_call.S, atm_call.K,
                                        atm_call.T, atm_call.r)
        assert result["parity_holds"]

    def test_parity_difference_reported(self):
        result = put_call_parity_check(5.0, 3.0, 100, 100, 1.0, 0.05)
        assert "difference" in result


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_spot_raises(self):
        with pytest.raises(ValueError):
            OptionParameters(S=-10, K=100, T=1, r=0.05, sigma=0.20)

    def test_invalid_option_type(self):
        with pytest.raises(ValueError):
            OptionParameters(S=100, K=100, T=1, r=0.05, sigma=0.20, option_type="straddle")

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            OptionParameters(S=100, K=100, T=1, r=0.05, sigma=-0.1)


# ---------------------------------------------------------------------------
# Analysis functions tests
# ---------------------------------------------------------------------------

class TestAnalysisFunctions:
    def test_greeks_vs_spot_shape(self, atm_call):
        df = greeks_vs_spot(atm_call, n_points=50)
        assert len(df) == 50
        assert "delta" in df.columns

    def test_greeks_vs_time_decreasing_price(self, atm_call):
        """Option price should decrease as time approaches expiry (for ATM)."""
        df = greeks_vs_time(atm_call, n_points=50)
        # Price should be highest at start (most time) and lowest near expiry
        first_price = df.iloc[0]["option_price"]
        last_price = df.iloc[-1]["option_price"]
        assert first_price > last_price

    def test_payoff_diagram_call_shape(self, atm_call):
        df = payoff_diagram(atm_call)
        # Below strike, P&L should be negative (lost premium)
        below_strike = df[df["spot_at_expiry"] < atm_call.K]["pnl"]
        assert (below_strike < 0).all()

    def test_option_summary_keys(self, atm_call):
        s = option_summary(atm_call)
        for key in ["option_price", "delta", "gamma", "theta", "vega", "rho"]:
            assert key in s

    def test_delta_hedge_simulation_runs(self, atm_call):
        df = delta_hedge_simulation(atm_call, n_steps=50, seed=1)
        assert len(df) == 51
        assert "hedge_error" in df.columns


# ---------------------------------------------------------------------------
# Volatility Surface tests
# ---------------------------------------------------------------------------

class TestVolatilitySurface:
    def test_iv_smile_shape(self):
        df = iv_smile(S=100, T=0.25, r=0.05, sigma_atm=0.20, skew=-0.10)
        assert "implied_vol_pct" in df.columns
        assert len(df) > 0
        assert (df["implied_vol_pct"] > 0).all()

    def test_vol_surface_shape(self):
        df = volatility_surface(S=100, r=0.05, sigma_atm=0.20)
        assert "maturity_years" in df.columns
        assert "implied_vol_pct" in df.columns
        assert len(df) > 0

    def test_skew_metrics(self):
        smile = iv_smile(S=100, T=0.25, r=0.05, sigma_atm=0.20, skew=-0.10)
        metrics = skew_metrics(smile, S=100)
        assert "atm_iv_pct" in metrics
        assert metrics["put_call_skew"] > 0  # puts more expensive with negative skew
