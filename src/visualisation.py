"""
visualisation.py
================
Publication-quality plots for the Black-Scholes toolkit.

All functions return matplotlib Figure objects so they can be displayed,
saved, or embedded in reports.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings

# Use a clean, professional style
plt.rcParams.update({
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#f5f5f5",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

COLORS = {
    "call": "#2196F3",
    "put": "#F44336",
    "neutral": "#4CAF50",
    "warning": "#FF9800",
    "dark": "#212121",
    "light": "#BDBDBD",
}


# ---------------------------------------------------------------------------
# 1. Option Price & Greeks vs Spot
# ---------------------------------------------------------------------------

def plot_greeks_vs_spot(df: pd.DataFrame, params, save_path: str = None) -> plt.Figure:
    """
    6-panel plot: Price, Delta, Gamma, Theta, Vega, Rho vs Spot.

    Parameters
    ----------
    df : pd.DataFrame
        Output of greeks_analysis.greeks_vs_spot().
    params : OptionParameters
        Used to draw reference lines at current spot and strike.
    save_path : str, optional
        If provided, save figure to this path.
    """
    color = COLORS["call"] if params.option_type == "call" else COLORS["put"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Black-Scholes Greeks vs Spot Price\n"
        f"{params.option_type.upper()} | K={params.K} | T={params.T:.2f}y | "
        f"σ={params.sigma*100:.1f}% | r={params.r*100:.1f}%"
    )

    plots = [
        ("option_price", "Option Price (€)", True),
        ("delta", "Delta (Δ)", False),
        ("gamma", "Gamma (Γ)", False),
        ("theta", "Theta (Θ) per day", False),
        ("vega", "Vega (ν) per 1% vol", False),
        ("rho", "Rho (ρ) per 1% rate", False),
    ]

    for ax, (col, ylabel, show_intrinsic) in zip(axes.flat, plots):
        ax.plot(df["spot"], df[col], color=color, linewidth=2, label=col.capitalize())

        if show_intrinsic:
            ax.plot(df["spot"], df["intrinsic_value"], "--", color=COLORS["light"],
                    linewidth=1.5, label="Intrinsic Value")
            ax.fill_between(df["spot"], df["intrinsic_value"], df["option_price"],
                            alpha=0.2, color=color, label="Time Value")

        ax.axvline(params.S, color=COLORS["dark"], linestyle=":", linewidth=1.2,
                   label=f"S={params.S}")
        ax.axvline(params.K, color=COLORS["warning"], linestyle="--", linewidth=1.2,
                   label=f"K={params.K}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Spot Price (S)")
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Payoff Diagram
# ---------------------------------------------------------------------------

def plot_payoff(df: pd.DataFrame, params, save_path: str = None) -> plt.Figure:
    """
    Option payoff and P&L diagram at expiry.

    Parameters
    ----------
    df : pd.DataFrame
        Output of greeks_analysis.payoff_diagram().
    """
    color = COLORS["call"] if params.option_type == "call" else COLORS["put"]
    breakeven = df["breakeven"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Payoff Diagram at Expiry – {params.option_type.upper()} Option\n"
        f"K={params.K} | Premium={params.S:.2f} | Breakeven={breakeven:.2f}"
    )

    ax.plot(df["spot_at_expiry"], df["pnl"], color=color, linewidth=2.5, label="P&L")
    ax.axhline(0, color=COLORS["dark"], linewidth=1, linestyle="-")
    ax.axvline(params.K, color=COLORS["warning"], linewidth=1.5, linestyle="--", label=f"Strike K={params.K}")
    ax.axvline(breakeven, color=COLORS["neutral"], linewidth=1.5, linestyle="--",
               label=f"Breakeven={breakeven:.2f}")
    ax.axvline(params.S, color=COLORS["light"], linewidth=1, linestyle=":", label=f"Current S={params.S}")

    # Shade profitable zone
    ax.fill_between(df["spot_at_expiry"], df["pnl"], 0,
                    where=df["pnl"] >= 0, alpha=0.25, color=COLORS["neutral"], label="Profit zone")
    ax.fill_between(df["spot_at_expiry"], df["pnl"], 0,
                    where=df["pnl"] < 0, alpha=0.25, color=COLORS["put"], label="Loss zone")

    ax.set_xlabel("Spot Price at Expiry")
    ax.set_ylabel("P&L (€)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. IV Smile
# ---------------------------------------------------------------------------

def plot_iv_smile(smile_df: pd.DataFrame, T: float, save_path: str = None) -> plt.Figure:
    """
    Plot the implied volatility smile for a given maturity.

    Parameters
    ----------
    smile_df : pd.DataFrame
        Output of volatility_surface.iv_smile().
    T : float
        Maturity in years (for title).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Implied Volatility Smile — Maturity T={T:.2f}y ({round(T*12)}M)")

    # Left: IV vs Strike
    axes[0].plot(smile_df["strike"], smile_df["implied_vol_pct"],
                 color=COLORS["call"], linewidth=2.5)
    axes[0].set_xlabel("Strike (K)")
    axes[0].set_ylabel("Implied Volatility (%)")
    axes[0].set_title("IV vs Strike")

    # Right: IV vs Moneyness
    axes[1].plot(smile_df["moneyness_pct"], smile_df["implied_vol_pct"],
                 color=COLORS["put"], linewidth=2.5)
    axes[1].axvline(0, color=COLORS["dark"], linestyle="--", linewidth=1, label="ATM")
    axes[1].set_xlabel("Moneyness (K-S)/S (%)")
    axes[1].set_ylabel("Implied Volatility (%)")
    axes[1].set_title("IV vs Moneyness (Smile / Skew)")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Volatility Surface (3-D)
# ---------------------------------------------------------------------------

def plot_volatility_surface(surface_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    3-D surface plot and contour heatmap of the implied volatility surface.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Output of volatility_surface.volatility_surface().
    """
    pivoted = surface_df.pivot_table(
        index="maturity_years", columns="strike", values="implied_vol_pct"
    )
    X = pivoted.columns.values      # strikes
    Y = pivoted.index.values        # maturities
    Z = pivoted.values              # IV surface

    XX, YY = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Implied Volatility Surface (Volatility Smile + Term Structure)")

    # Left: 3-D surface
    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(XX, YY, Z, cmap="RdYlGn_r", alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax3d, shrink=0.5, label="IV (%)")
    ax3d.set_xlabel("Strike (K)")
    ax3d.set_ylabel("Maturity (years)")
    ax3d.set_zlabel("Implied Vol (%)")
    ax3d.set_title("3-D Volatility Surface")
    ax3d.view_init(elev=30, azim=-60)

    # Right: heatmap
    ax2d = fig.add_subplot(122)
    im = ax2d.contourf(XX, YY, Z, levels=20, cmap="RdYlGn_r")
    fig.colorbar(im, ax=ax2d, label="IV (%)")
    ax2d.set_xlabel("Strike (K)")
    ax2d.set_ylabel("Maturity (years)")
    ax2d.set_title("Volatility Surface Heatmap")

    # Add contour lines
    cs = ax2d.contour(XX, YY, Z, levels=10, colors="white", alpha=0.4, linewidths=0.8)
    ax2d.clabel(cs, inline=True, fontsize=7, fmt="%.1f%%")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Theta Decay
# ---------------------------------------------------------------------------

def plot_theta_decay(df: pd.DataFrame, params, save_path: str = None) -> plt.Figure:
    """
    Plot option price decay over time (Theta visualisation).

    Parameters
    ----------
    df : pd.DataFrame
        Output of greeks_analysis.greeks_vs_time().
    """
    color = COLORS["call"] if params.option_type == "call" else COLORS["put"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Theta Decay — {params.option_type.upper()} Option\n"
        f"S={params.S} | K={params.K} | σ={params.sigma*100:.1f}% | r={params.r*100:.1f}%"
    )

    axes[0].plot(df["days_to_expiry"], df["option_price"], color=color, linewidth=2.5)
    axes[0].set_xlabel("Days to Expiry")
    axes[0].set_ylabel("Option Price (€)")
    axes[0].set_title("Option Price vs Time to Expiry")
    axes[0].invert_xaxis()
    axes[0].axhline(0, color=COLORS["light"], linestyle="--", linewidth=1)

    axes[1].plot(df["days_to_expiry"], df["theta"], color=COLORS["warning"], linewidth=2.5)
    axes[1].set_xlabel("Days to Expiry")
    axes[1].set_ylabel("Theta (€/day)")
    axes[1].set_title("Theta (Daily Time Decay)")
    axes[1].invert_xaxis()
    axes[1].axhline(0, color=COLORS["light"], linestyle="--", linewidth=1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Delta-Hedging Simulation
# ---------------------------------------------------------------------------

def plot_delta_hedge(sim_df: pd.DataFrame, params, save_path: str = None) -> plt.Figure:
    """
    Visualise a Delta-Hedging simulation.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Output of greeks_analysis.delta_hedge_simulation().
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Delta-Hedging Simulation — {params.option_type.upper()} Option\n"
        f"S₀={params.S} | K={params.K} | T={params.T:.2f}y | "
        f"σ={params.sigma*100:.1f}% | r={params.r*100:.1f}%"
    )

    # (1) Simulated stock path
    axes[0, 0].plot(sim_df["day"], sim_df["S"], color=COLORS["call"], linewidth=1.8)
    axes[0, 0].axhline(params.K, color=COLORS["warning"], linestyle="--", linewidth=1,
                        label=f"Strike K={params.K}")
    axes[0, 0].set_title("Simulated Stock Path")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Stock Price (€)")
    axes[0, 0].legend()

    # (2) Delta (shares held) over time
    axes[0, 1].plot(sim_df["day"], sim_df["delta"], color=COLORS["neutral"], linewidth=1.8,
                    label="Required Delta")
    axes[0, 1].plot(sim_df["day"], sim_df["shares_held"], "--", color=COLORS["warning"],
                    linewidth=1.2, alpha=0.7, label="Shares Held")
    axes[0, 1].set_title("Delta & Hedge Position Over Time")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("Delta / Shares")
    axes[0, 1].legend()

    # (3) Option price vs portfolio value
    axes[1, 0].plot(sim_df["day"], sim_df["option_price"], color=COLORS["put"],
                    linewidth=2, label="Option Price (BS)")
    axes[1, 0].plot(sim_df["day"], sim_df["portfolio_value"], "--", color=COLORS["call"],
                    linewidth=1.5, alpha=0.8, label="Hedge Portfolio")
    axes[1, 0].set_title("Option Price vs Hedge Portfolio Value")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Value (€)")
    axes[1, 0].legend()

    # (4) Hedging error over time
    axes[1, 1].fill_between(sim_df["day"], sim_df["hedge_error"], 0,
                             alpha=0.5, color=COLORS["warning"])
    axes[1, 1].plot(sim_df["day"], sim_df["hedge_error"], color=COLORS["warning"],
                    linewidth=1.5, label="Hedge Error")
    axes[1, 1].axhline(0, color=COLORS["dark"], linewidth=1)
    axes[1, 1].set_title("Hedging Error (Portfolio − Option Price)")
    axes[1, 1].set_xlabel("Day")
    axes[1, 1].set_ylabel("Error (€)")
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Dashboard Overview
# ---------------------------------------------------------------------------

def plot_dashboard(params, save_path: str = None) -> plt.Figure:
    """
    Single-page summary dashboard for one option.
    Includes: Payoff, Price vs Spot, Delta vs Spot, Gamma vs Spot, 
              Theta Decay, Vega vs Vol.
    """
    from .greeks_analysis import (greeks_vs_spot, greeks_vs_time, 
                                   greeks_vs_volatility, payoff_diagram, option_summary)

    summary = option_summary(params)
    df_spot = greeks_vs_spot(params)
    df_time = greeks_vs_time(params)
    df_vol  = greeks_vs_volatility(params)
    df_pay  = payoff_diagram(params)

    color = COLORS["call"] if params.option_type == "call" else COLORS["put"]
    breakeven = df_pay["breakeven"].iloc[0]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Black-Scholes Dashboard — {params.option_type.upper()} Option\n"
        f"Price: {summary['option_price']:.4f} | "
        f"Δ={summary['delta']:.4f} | Γ={summary['gamma']:.6f} | "
        f"Θ={summary['theta']:.4f}/day | ν={summary['vega']:.4f}/1%vol"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Payoff
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_pay["spot_at_expiry"], df_pay["pnl"], color=color, linewidth=2)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axvline(params.K, color=COLORS["warning"], linestyle="--", linewidth=1)
    ax1.fill_between(df_pay["spot_at_expiry"], df_pay["pnl"], 0,
                     where=df_pay["pnl"] >= 0, alpha=0.2, color=COLORS["neutral"])
    ax1.fill_between(df_pay["spot_at_expiry"], df_pay["pnl"], 0,
                     where=df_pay["pnl"] < 0, alpha=0.2, color=COLORS["put"])
    ax1.set_title(f"Payoff at Expiry (Breakeven: {breakeven:.2f})")
    ax1.set_xlabel("Spot at Expiry")
    ax1.set_ylabel("P&L (€)")

    # Price vs Spot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_spot["spot"], df_spot["option_price"], color=color, linewidth=2)
    ax2.plot(df_spot["spot"], df_spot["intrinsic_value"], "--", color=COLORS["light"], linewidth=1.5)
    ax2.axvline(params.S, color=COLORS["dark"], linestyle=":", linewidth=1)
    ax2.set_title("Option Price vs Spot")
    ax2.set_xlabel("Spot (S)")
    ax2.set_ylabel("Price (€)")

    # Delta vs Spot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df_spot["spot"], df_spot["delta"], color=COLORS["neutral"], linewidth=2)
    ax3.axvline(params.S, color=COLORS["dark"], linestyle=":", linewidth=1)
    ax3.axvline(params.K, color=COLORS["warning"], linestyle="--", linewidth=1)
    ax3.set_title("Delta vs Spot")
    ax3.set_xlabel("Spot (S)")
    ax3.set_ylabel("Delta (Δ)")

    # Gamma vs Spot
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(df_spot["spot"], df_spot["gamma"], color=COLORS["put"], linewidth=2)
    ax4.axvline(params.K, color=COLORS["warning"], linestyle="--", linewidth=1)
    ax4.set_title("Gamma vs Spot (highest ATM)")
    ax4.set_xlabel("Spot (S)")
    ax4.set_ylabel("Gamma (Γ)")

    # Theta decay
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(df_time["days_to_expiry"], df_time["option_price"], color=COLORS["warning"], linewidth=2)
    ax5.invert_xaxis()
    ax5.set_title("Time Decay (Theta)")
    ax5.set_xlabel("Days to Expiry →")
    ax5.set_ylabel("Option Price (€)")

    # Vega vs Vol
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(df_vol["volatility_pct"], df_vol["option_price"], color=COLORS["call"], linewidth=2)
    ax6.axvline(params.sigma * 100, color=COLORS["dark"], linestyle=":", linewidth=1,
                label=f"Current σ={params.sigma*100:.1f}%")
    ax6.set_title("Option Price vs Volatility")
    ax6.set_xlabel("Volatility σ (%)")
    ax6.set_ylabel("Option Price (€)")
    ax6.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
