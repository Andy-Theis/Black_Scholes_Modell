#!/usr/bin/env python3
"""
main.py
=======
Interactive command-line interface for the Black-Scholes toolkit.

Usage:
    python main.py                  # Interactive mode
    python main.py --demo           # Run full demo with all plots
    python main.py --price          # Price a single option
    python main.py --surface        # Generate volatility surface
    python main.py --hedge          # Run delta-hedge simulation

Author: [Your Name]
"""

import argparse
import os
import sys
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.black_scholes import (
    OptionParameters, price, all_greeks,
    implied_volatility, put_call_parity_check
)
from src.greeks_analysis import (
    greeks_vs_spot, greeks_vs_time, payoff_diagram,
    delta_hedge_simulation, option_summary
)
from src.volatility_surface import iv_smile, volatility_surface, skew_metrics
from src import visualisation

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

def print_header(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_dict(d: dict, indent: int = 2):
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{pad}{k:35s}: {v:.6f}")
        else:
            print(f"{pad}{k:35s}: {v}")


# ---------------------------------------------------------------------------
# Demo: run everything
# ---------------------------------------------------------------------------

def run_demo():
    """Full demonstration of all toolkit features."""

    print_header("BLACK-SCHOLES TOOLKIT — FULL DEMO")

    # -------- 1. Option pricing ----------------------------------------
    print_header("1. OPTION PRICING")

    call_params = OptionParameters(
        S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call"
    )
    put_params = OptionParameters(
        S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="put"
    )

    print("\n  CALL Option Summary:")
    print_dict(option_summary(call_params))

    print("\n  PUT Option Summary:")
    print_dict(option_summary(put_params))

    # -------- 2. Put-Call Parity ----------------------------------------
    print_header("2. PUT-CALL PARITY CHECK")
    pcp = put_call_parity_check(
        price(call_params), price(put_params),
        call_params.S, call_params.K, call_params.T, call_params.r
    )
    print_dict(pcp)

    # -------- 3. Implied Volatility ------------------------------------
    print_header("3. IMPLIED VOLATILITY")
    market_price = 3.50  # hypothetical observed market price
    iv = implied_volatility(
        market_price=market_price,
        S=call_params.S, K=call_params.K, T=call_params.T,
        r=call_params.r, option_type="call"
    )
    print(f"  Market price: {market_price}")
    print(f"  BS theoretical price: {price(call_params):.4f}")
    print(f"  Implied Volatility: {iv*100:.2f}%  (model σ={call_params.sigma*100:.1f}%)")

    # -------- 4. Greeks dashboard --------------------------------------
    print_header("4. GREEKS DASHBOARD — Saving plot...")
    fig = visualisation.plot_dashboard(call_params, save_path=f"{OUTPUT_DIR}/dashboard.png")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/dashboard.png")

    # -------- 5. Greeks vs Spot ----------------------------------------
    print_header("5. GREEKS VS SPOT — Saving plot...")
    df_spot = greeks_vs_spot(call_params)
    fig = visualisation.plot_greeks_vs_spot(df_spot, call_params,
                                             save_path=f"{OUTPUT_DIR}/greeks_vs_spot.png")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/greeks_vs_spot.png")

    # -------- 6. Payoff diagram ----------------------------------------
    print_header("6. PAYOFF DIAGRAM — Saving plot...")
    df_pay = payoff_diagram(call_params)
    fig = visualisation.plot_payoff(df_pay, call_params,
                                    save_path=f"{OUTPUT_DIR}/payoff.png")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/payoff.png")

    # -------- 7. Theta decay -------------------------------------------
    print_header("7. THETA DECAY — Saving plot...")
    atm_call = OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    df_time = greeks_vs_time(atm_call)
    fig = visualisation.plot_theta_decay(df_time, atm_call,
                                          save_path=f"{OUTPUT_DIR}/theta_decay.png")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/theta_decay.png")

    # -------- 8. IV Smile ---------------------------------------------
    print_header("8. VOLATILITY SMILE — Saving plot...")
    smile_df = iv_smile(S=100, T=0.25, r=0.05, sigma_atm=0.20, skew=-0.10)
    metrics = skew_metrics(smile_df, S=100)
    print("  Smile metrics:")
    print_dict(metrics)
    fig = visualisation.plot_iv_smile(smile_df, T=0.25,
                                       save_path=f"{OUTPUT_DIR}/iv_smile.png")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/iv_smile.png")

    # -------- 9. Volatility Surface ------------------------------------
    print_header("9. VOLATILITY SURFACE — Saving plot...")
    surf_df = volatility_surface(S=100, r=0.05, sigma_atm=0.20, skew=-0.10)
    fig = visualisation.plot_volatility_surface(surf_df,
                                                 save_path=f"{OUTPUT_DIR}/vol_surface.png")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/vol_surface.png")

    # -------- 10. Delta-Hedge Simulation -------------------------------
    print_header("10. DELTA-HEDGE SIMULATION — Saving plot...")
    hedge_params = OptionParameters(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    sim_df = delta_hedge_simulation(hedge_params, n_steps=252, seed=42)
    final_error = sim_df["hedge_error"].iloc[-1]
    max_error = sim_df["hedge_error"].abs().max()
    print(f"  Final hedge error: {final_error:.4f}")
    print(f"  Max absolute hedge error: {max_error:.4f}")
    fig = visualisation.plot_delta_hedge(sim_df, hedge_params,
                                          save_path=f"{OUTPUT_DIR}/delta_hedge.png")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/delta_hedge.png")

    print_header("DEMO COMPLETE")
    print(f"\n  All outputs saved to '{OUTPUT_DIR}/' directory.\n")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode():
    """Prompt user for parameters and compute price + Greeks."""
    print_header("BLACK-SCHOLES CALCULATOR — INTERACTIVE MODE")

    print("\nEnter option parameters (press Enter for defaults):\n")

    def get_float(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return float(val) if val else default

    def get_str(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip().lower()
        return val if val in ("call", "put") else default

    S       = get_float("Spot price (S)", 100.0)
    K       = get_float("Strike price (K)", 100.0)
    T       = get_float("Time to expiry in years (T)", 0.25)
    r       = get_float("Risk-free rate (e.g. 0.05 = 5%)", 0.05)
    sigma   = get_float("Volatility (e.g. 0.20 = 20%)", 0.20)
    q       = get_float("Dividend yield (e.g. 0.02 = 2%)", 0.0)
    opt_type = get_str("Option type (call/put)", "call")

    params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=opt_type)

    print("\n")
    print_header("RESULTS")
    print_dict(option_summary(params))

    print("\n  Would you like to save plots? (y/n)")
    save = input("  > ").strip().lower() == "y"

    if save:
        matplotlib.use("Agg")
        path = f"{OUTPUT_DIR}/custom_option.png"
        fig = visualisation.plot_dashboard(params, save_path=path)
        plt.close(fig)
        print(f"\n  Dashboard saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Black-Scholes Options Toolkit",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run full demonstration (all features + plots)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter parameters interactively"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run unit tests (equivalent to: pytest tests/)"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.interactive:
        interactive_mode()
    elif args.test:
        import subprocess
        subprocess.run(["pytest", "tests/", "-v"])
    else:
        # Default: show quick example + menu
        print_header("BLACK-SCHOLES TOOLKIT")
        print("\n  Usage:")
        print("    python main.py --demo          Run full demo")
        print("    python main.py --interactive   Interactive calculator")
        print("    python main.py --test          Run unit tests")
        print("\n  Quick example:\n")

        p = OptionParameters(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call")
        print_dict(option_summary(p))


if __name__ == "__main__":
    main()
