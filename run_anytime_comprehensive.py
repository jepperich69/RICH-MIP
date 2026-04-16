"""
run_anytime_comprehensive.py
============================
Generates the data for Figures 1 & 2 (time-quality trajectories and
relative-advantage envelopes) in the RICH-MIP paper.

Design
------
Scales  : 200×4000, 250×5000, 300×6000
Trials  : 20 per scale (seeds 42 … 61)
RICH    : full Stages 1-4 pipeline (IPF + rounding + restricted-MIP polish + MH)
Gurobi  : 30 s time-limited, single-threaded, anytime callback recording
          every new incumbent

Output
------
experiments/anytime_comprehensive/anytime_comp_<timestamp>.csv

Schema (one row per trial × checkpoint):
  n, m, trial, seed,
  checkpoint,          # time (s) at which objective is evaluated
  rich_obj,            # RICH objective at this checkpoint (NaN if not yet finished)
  gurobi_obj,          # Gurobi best incumbent at this checkpoint (NaN if none yet)
  rich_finish_time,    # wall time when RICH completed (s)
  rich_feasible        # 1 / 0

Usage
-----
  python run_anytime_comprehensive.py
  python run_anytime_comprehensive.py --trials 5   # quick smoke-test
"""

import argparse
import contextlib
import datetime
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mip_hybrid.apps.synth_setcover import (
    _gen_entropy_friendly_scp,
    _build_instance_from_matrix,
    solve_entropy_setcover,
)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
SCALES     = [(100, 3000), (200, 4000), (300, 5000)]
N_TRIALS   = 20
BASE_SEED  = 42

# RICH pipeline
TAU_SCHEDULE = "0.5,0.2,0.1"
ITERS        = 60
TOL          = 5e-4
POLISH_TIME  = 1.0
POLISH_POOL  = 0.3
DO_MH        = True
MH_STEPS     = 150

# Gurobi
MIP_TIMELIMIT = 30.0

# Time checkpoints at which objectives are recorded
CHECKPOINTS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def forward_fill(trace, checkpoints):
    """
    Given a list of (t, obj) event pairs, return the best incumbent seen
    at or before each checkpoint.  Returns NaN for checkpoints that precede
    the first incumbent.
    """
    result = []
    current = float("nan")
    trace_sorted = sorted(trace, key=lambda x: x[0])
    idx = 0
    for cp in checkpoints:
        while idx < len(trace_sorted) and trace_sorted[idx][0] <= cp:
            current = float(trace_sorted[idx][1])
            idx += 1
        result.append(current)
    return result


@contextlib.contextmanager
def suppress_stdout():
    """Redirect stdout to /dev/null for noisy solver output."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def run_gurobi_anytime(A, c, timelimit=30.0):
    """
    Solve the set-cover MIP with Gurobi for `timelimit` seconds, recording
    every new incumbent via the MIPSOL callback.

    Returns list of (runtime_s, obj_value) pairs.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("  [WARNING] gurobipy not available")
        return []

    n, m = A.shape
    log = []

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            rt  = model.cbGet(GRB.Callback.RUNTIME)
            log.append((float(rt), float(obj)))

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag",  0)
        env.setParam("TimeLimit",   float(timelimit))
        env.setParam("Threads",     1)
        env.start()

        with gp.Model(env=env) as mdl:
            x = mdl.addMVar(m, vtype=GRB.BINARY, name="x")
            for i in range(n):
                idx = np.where(A[i] == 1)[0]
                if idx.size:
                    mdl.addConstr(x[idx].sum() >= 1.0)
            mdl.setObjective(c @ x, GRB.MINIMIZE)
            mdl.optimize(cb)

    return log


def run_trial(n, m, seed):
    """
    Run one trial: RICH (Stages 1-4) + Gurobi anytime.

    Returns
    -------
    rich_obj   : float
    rich_time  : float  (wall-clock seconds for RICH to finish)
    rich_feas  : bool
    gurobi_log : list of (t, obj)
    """
    A, c = _gen_entropy_friendly_scp(n, m, seed=seed)

    # RICH
    t0 = time.time()
    with suppress_stdout():
        x_int, rich_obj, rich_feas, _ = solve_entropy_setcover(
            A, c,
            tau=0.1,
            iters=ITERS,
            tol=TOL,
            tau_schedule=TAU_SCHEDULE,
            polish_time=POLISH_TIME,
            polish_pool=POLISH_POOL,
            do_polish_mh=DO_MH,
            mh_steps=MH_STEPS,
            mh_seed=seed,
        )
    rich_time = time.time() - t0

    # Gurobi anytime
    gurobi_log = run_gurobi_anytime(A, c, timelimit=MIP_TIMELIMIT)

    return float(rich_obj), rich_time, bool(rich_feas), gurobi_log


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main(n_trials=N_TRIALS):
    out_dir = HERE / "experiments" / "anytime_comprehensive"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = out_dir / f"anytime_comp_{stamp}.csv"

    rows = []
    seeds = [BASE_SEED + i for i in range(n_trials)]

    for n, m in SCALES:
        print(f"\n=== {n}×{m} ===")
        for trial_idx, seed in enumerate(seeds):
            t_wall = time.time()
            print(f"  trial {trial_idx+1:2d}/{n_trials} (seed={seed}) ...", end=" ", flush=True)

            rich_obj, rich_time, rich_feas, grb_log = run_trial(n, m, seed)

            # RICH trace: single event at rich_time, flat thereafter
            rich_trace    = [(rich_time, rich_obj)] if rich_feas else []
            rich_at_cp    = forward_fill(rich_trace,  CHECKPOINTS)
            gurobi_at_cp  = forward_fill(grb_log,     CHECKPOINTS)

            grb_final = grb_log[-1][1] if grb_log else float("nan")
            print(f"RICH={rich_obj:.2f} @ {rich_time:.2f}s   GRB_30s={grb_final:.2f}"
                  f"   ({time.time()-t_wall:.1f}s wall)")

            for cp_idx, cp in enumerate(CHECKPOINTS):
                rows.append({
                    "n":               n,
                    "m":               m,
                    "trial":           trial_idx,
                    "seed":            seed,
                    "checkpoint":      cp,
                    "rich_obj":        rich_at_cp[cp_idx],
                    "gurobi_obj":      gurobi_at_cp[cp_idx],
                    "rich_finish_time": rich_time,
                    "rich_feasible":   int(rich_feas),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}  ({len(df)} rows, {len(rows)//len(CHECKPOINTS)} trials total)")

    # Generate figures → Overleaf_source/
    print("\nGenerating figures ...")
    import importlib.util as _ilu
    _fig_script = HERE.parents[2] / "make_figures_comprehensive.py"
    _spec = _ilu.spec_from_file_location("make_figures_comprehensive", _fig_script)
    _fig_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fig_mod)
    _fig_mod.main(csv_path=str(out_csv))

    return out_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=N_TRIALS,
                   help=f"Trials per scale (default: {N_TRIALS})")
    args = p.parse_args()
    main(n_trials=args.trials)
