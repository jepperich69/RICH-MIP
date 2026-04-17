"""
make_figures_comprehensive.py
==============================
Generates Figures 1 & 2 for the RICH-MIP paper from the anytime_comp CSV
produced by run_anytime_comprehensive.py.

Figure 1  (time_quality_figure.png / .svg)
  3 side-by-side panels — one per scale.
  Each panel: 20 individual trajectories (thin, semi-transparent) + bold median
  + IQR shaded band for RICH (green) and Gurobi (red).
  y-axis: gap relative to the best objective found by either method in that trial (%).

Figure 2  (time_advantage_figure.png / .svg)
  3 side-by-side panels.
  y-axis: relative advantage of RICH over Gurobi (%) = 100*(GRB_obj - RICH_obj)/RICH_obj.
  Positive = RICH is better; negative = Gurobi is better.
  Same envelope treatment as Figure 1.

Usage
-----
  python make_figures_comprehensive.py                   # uses latest CSV
  python make_figures_comprehensive.py --csv path/to.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCALES      = [(100, 3000), (200, 4000), (300, 5000)]
CHECKPOINTS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0]

C_RICH  = "#1b5e20"   # dark green
C_GRB   = "#b71c1c"   # dark red
C_ADV   = "#1f4e79"   # dark blue

ALPHA_TRACE  = 0.12   # individual trial lines
ALPHA_BAND   = 0.20   # IQR shaded region
LW_MEDIAN    = 2.4
LW_TRACE     = 0.8

HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "paper_artifacts"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_latest_csv():
    exp_dir = HERE / "experiments" / "anytime_comprehensive"
    csvs = sorted(exp_dir.glob("anytime_comp_*.csv"))
    if not csvs:
        sys.exit("No anytime_comp CSV found. Run run_anytime_comprehensive.py first.")
    path = csvs[-1]
    print(f"Loading {path}")
    return pd.read_csv(path)


def scale_trials(df, n, m):
    """Return sub-DataFrame for one scale, sorted by trial then checkpoint."""
    return df[(df["n"] == n) & (df["m"] == m)].sort_values(["trial", "checkpoint"])


def per_trial_gap(sub, col):
    """
    Convert raw objectives to gap (%) relative to the best objective found
    by either method in the same trial.

    Returns a dict {trial_id: [gap_at_cp0, gap_at_cp1, ...]}
    """
    result = {}
    for trial, grp in sub.groupby("trial"):
        grp = grp.sort_values("checkpoint")
        r_vals = grp["rich_obj"].values
        g_vals = grp["gurobi_obj"].values
        # Best = minimum finite value across both methods in this trial
        all_vals = np.concatenate([r_vals[np.isfinite(r_vals)],
                                   g_vals[np.isfinite(g_vals)]])
        if all_vals.size == 0:
            result[trial] = [np.nan] * len(CHECKPOINTS)
            continue
        ref = all_vals.min()
        gaps = []
        for v in grp[col].values:
            gaps.append(100.0 * (v - ref) / ref if np.isfinite(v) else np.nan)
        result[trial] = gaps
    return result


def per_trial_advantage(sub):
    """
    Relative advantage of RICH over Gurobi:
      adv = 100 * (gurobi_obj - rich_obj) / rich_obj
    Positive = RICH is better (lower cost); negative = Gurobi better.
    Returns dict {trial_id: [adv_at_cp0, ...]}
    """
    result = {}
    for trial, grp in sub.groupby("trial"):
        grp = grp.sort_values("checkpoint")
        r_vals = grp["rich_obj"].values
        g_vals = grp["gurobi_obj"].values
        advs = []
        for r, g in zip(r_vals, g_vals):
            if np.isfinite(r) and np.isfinite(g) and r > 1e-9:
                advs.append(100.0 * (g - r) / r)
            else:
                advs.append(np.nan)
        result[trial] = advs
    return result


def envelope(by_trial):
    """
    Given {trial: [val_per_cp]}, return (median, q25, q75) arrays over trials
    at each checkpoint.
    """
    n_cp = len(CHECKPOINTS)
    matrix = np.full((len(by_trial), n_cp), np.nan)
    for i, vals in enumerate(by_trial.values()):
        matrix[i, :] = vals

    med = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    return med, q25, q75, matrix


# ---------------------------------------------------------------------------
# Figure 1 — time-quality trajectories
# ---------------------------------------------------------------------------

def make_figure1(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=False)

    for ax, (n, m) in zip(axes, SCALES):
        sub = scale_trials(df, n, m)

        rich_gaps = per_trial_gap(sub, "rich_obj")
        grb_gaps  = per_trial_gap(sub, "gurobi_obj")

        r_med, r_q25, r_q75, r_mat = envelope(rich_gaps)
        g_med, g_q25, g_q75, g_mat = envelope(grb_gaps)

        cps = np.array(CHECKPOINTS)

        # Individual trial lines
        for row in r_mat:
            ax.plot(cps, row, color=C_RICH, alpha=ALPHA_TRACE, linewidth=LW_TRACE)
        for row in g_mat:
            ax.plot(cps, row, color=C_GRB,  alpha=ALPHA_TRACE, linewidth=LW_TRACE)

        # IQR bands
        ax.fill_between(cps, r_q25, r_q75, alpha=ALPHA_BAND, color=C_RICH)
        ax.fill_between(cps, g_q25, g_q75, alpha=ALPHA_BAND, color=C_GRB)

        # Median lines
        ax.plot(cps, r_med, color=C_RICH, linewidth=LW_MEDIAN, label="RICH (median)")
        ax.plot(cps, g_med, color=C_GRB,  linewidth=LW_MEDIAN, label="Gurobi (median)")

        ax.axhline(0, color="#999", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(f"$n={n},\\ m={m}$", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_xlim(0, 30)
        ax.set_ylim(bottom=-0.5)
        ax.grid(True, alpha=0.22)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    axes[0].set_ylabel("Gap to trial-best objective (%)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    for ext in ("png", "svg"):
        path = out_dir / f"time_quality_figure.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — relative advantage
# ---------------------------------------------------------------------------

def make_figure2(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=False)

    cps = np.array(CHECKPOINTS)

    for ax, (n, m) in zip(axes, SCALES):
        sub   = scale_trials(df, n, m)
        advs  = per_trial_advantage(sub)
        med, q25, q75, mat = envelope(advs)

        # Individual trial lines
        for row in mat:
            ax.plot(cps, row, color=C_ADV, alpha=ALPHA_TRACE, linewidth=LW_TRACE)

        # IQR band
        ax.fill_between(cps, q25, q75, alpha=ALPHA_BAND, color=C_ADV)

        # Median line
        ax.plot(cps, med, color=C_ADV, linewidth=LW_MEDIAN)

        # Zero reference
        ax.axhline(0, color="#444", linewidth=1.2, linestyle="--", zorder=3)

        # Annotations
        ax.text(0.97, 0.93, "RICH better", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=C_RICH)
        ax.text(0.97, 0.07, "Gurobi better", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=C_GRB)

        ax.set_title(f"$n={n},\\ m={m}$", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_xlim(0, 30)
        ax.grid(True, alpha=0.22)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    axes[0].set_ylabel("Relative advantage of RICH over Gurobi (%)", fontsize=11)

    fig.tight_layout()

    for ext in ("png", "svg"):
        path = out_dir / f"time_advantage_figure.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path}")
    else:
        df = load_latest_csv()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    make_figure1(df, ARTIFACT_DIR)
    make_figure2(df, ARTIFACT_DIR)
    print(f"\nDone. Figures saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None,
                   help="Path to anytime_comp CSV (default: latest in experiments/anytime_comprehensive/)")
    args = p.parse_args()
    main(args.csv)
