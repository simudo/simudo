#!/usr/bin/env python3
"""
Compare J(V) curves from two fourlayer simulation runs:
  - Runner output:    sim_V.csv  (from simudo_1d_runner.py)
  - Reference output: pd_V.csv   (from fourlayer_reference.py / fourlayer.py)

Usage:
    python3 gui/compare_jv.py <runner_dir> <reference_dir> [--save path/to/fig.png]

Defaults:
    runner_dir    = code/2026Apr27/d          (last completed runner run)
    reference_dir = gui/out/reference         (fourlayer_reference.py output)

The script requires only pandas, numpy, and matplotlib — no Simudo or dolfin.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_runner_jv(folder):
    """Read J-V from sim_V.csv produced by simudo_1d_runner.py.

    J = avg:current_CB:right_contact + avg:current_VB:right_contact  (mA/cm²)
    V = sweep_parameter:V  (V)
    """
    csv = Path(folder) / "sim_V.csv"
    # Row 0 = column names, row 1 = units comment (starts with '#'), row 2+ = data
    df = pd.read_csv(csv, comment="#", header=0)
    v = df["sweep_parameter:V"].values.astype(float)
    j = (df["avg:current_CB:right_contact"].values.astype(float)
         + df["avg:current_VB:right_contact"].values.astype(float))
    order = np.argsort(v)
    return v[order], j[order]


def read_reference_jv(folder):
    """Read J-V from pd_V.csv produced by fourlayer.py / fourlayer_reference.py.

    J = avg:current_CB:n_contact + avg:current_VB:n_contact  (mA/cm²)
    V = sweep_parameter:V  (V)
    """
    csv = Path(folder) / "pd_V.csv"
    df = pd.read_csv(csv, comment="#", header=0)
    v = df["sweep_parameter:V"].values.astype(float)
    j = (df["avg:current_CB:n_contact"].values.astype(float)
         + df["avg:current_VB:n_contact"].values.astype(float))
    order = np.argsort(v)
    return v[order], j[order]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def max_power_point(v, j):
    """Return (Vmpp, Jmpp, Pmax) where P = -J*V and J < 0 for generation."""
    p = -j * v          # positive power in power-producing quadrant
    idx = np.argmax(p)
    return v[idx], j[idx], p[idx]


def print_comparison_table(v_run, j_run, v_ref, j_ref):
    """Print a side-by-side table of J values at common voltages."""
    # Interpolate reference onto runner voltages
    j_ref_interp = np.interp(v_run, v_ref, j_ref)
    delta = j_run - j_ref_interp

    print(f"\n{'V (V)':>8}  {'J_runner':>14}  {'J_ref':>14}  {'ΔJ':>14}  {'Δ%':>8}")
    print("-" * 65)
    for v, jr, jref, dj in zip(v_run, j_run, j_ref_interp, delta):
        pct = 100 * dj / jref if abs(jref) > 1e-12 else float("nan")
        print(f"{v:8.4f}  {jr:14.4f}  {jref:14.4f}  {dj:14.4f}  {pct:8.3f}%")

    rms = np.sqrt(np.mean(delta**2))
    print(f"\nRMS ΔJ = {rms:.4f} mA/cm²")
    return rms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runner_dir",   nargs="?",
                        default="2026Apr27/d",
                        help="Folder containing sim_V.csv")
    parser.add_argument("reference_dir", nargs="?",
                        default="gui/out/reference",
                        help="Folder containing pd_V.csv")
    parser.add_argument("--save", default=None,
                        help="Save figure to this path (default: show interactively)")
    args = parser.parse_args()

    print(f"Runner output:    {args.runner_dir}")
    print(f"Reference output: {args.reference_dir}")

    try:
        v_run, j_run = read_runner_jv(args.runner_dir)
    except FileNotFoundError as e:
        sys.exit(f"ERROR reading runner output: {e}")
    try:
        v_ref, j_ref = read_reference_jv(args.reference_dir)
    except FileNotFoundError as e:
        sys.exit(f"ERROR reading reference output: {e}")

    # Max power points
    Vmpp_run, Jmpp_run, Pmax_run = max_power_point(v_run, j_run)
    Vmpp_ref, Jmpp_ref, Pmax_ref = max_power_point(v_ref, j_ref)
    print(f"\nRunner   MPP: V={Vmpp_run:.4f} V, J={Jmpp_run:.4f} mA/cm², P={Pmax_run:.4f} mW/cm²")
    print(f"Reference MPP: V={Vmpp_ref:.4f} V, J={Jmpp_ref:.4f} mA/cm², P={Pmax_ref:.4f} mW/cm²")

    rms = print_comparison_table(v_run, j_run, v_ref, j_ref)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax1.plot(v_ref, j_ref, "k-",  lw=2,   label="Reference (fourlayer.py)")
    ax1.plot(v_run, j_run, "r--", lw=1.5, label="Runner (simudo_1d_runner.py)")
    ax1.set_ylabel("J (mA/cm²)")
    ax1.legend()
    ax1.set_title("J–V comparison: runner vs. reference")
    ax1.grid(True, alpha=0.3)

    # Difference panel
    j_ref_interp = np.interp(v_run, v_ref, j_ref)
    ax2.plot(v_run, j_run - j_ref_interp, "b-", lw=1.5)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("V (V)")
    ax2.set_ylabel("ΔJ = J_runner − J_ref  (mA/cm²)")
    ax2.set_title(f"Difference  (RMS = {rms:.4f} mA/cm²)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = args.save or "gui/out/jv_comparison.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")


if __name__ == "__main__":
    main()
