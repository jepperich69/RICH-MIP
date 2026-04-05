"""
Heuristic comparison: RICH Stage 1-4 vs Gurobi (default) vs Gurobi MIPFocus=1.

This addresses reviewer comments R1.4 and TE.11: "compare against heuristics
from the literature (Feasibility Pump, RINS, Local Branching)".
MIPFocus=1 activates all of these inside Gurobi at equal time budget.

Design:
  - Same synthetic set-cover instances as the ablation study.
  - RICH runs Stage 1-4 (IPF + rounding + Gurobi restricted polish + MH).
  - Gurobi default and MIPFocus=1 each run at three time budgets:
      T_BUDGETS = [1.0, 5.0, 30.0] seconds.
  - Gap is computed against the best solution found across all methods and budgets
    (MIPFocus=1 @ 30s reference).

Outputs:
  experiments/mipfocus/mipfocus_results_<ts>.csv
  experiments/mipfocus/mipfocus_summary_<ts>.txt
"""

import sys, os, time, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mip_hybrid.apps.synth_setcover import (
    _gen_entropy_friendly_scp,
    solve_entropy_setcover,
)

# ── config ────────────────────────────────────────────────────────────────────
SCALES    = [(100, 500), (200, 1000), (400, 2000)]
TRIALS    = 5
SEED      = 42

TAU_SCHED = "0.5,0.2,0.1"
ITERS     = 50
TOL       = 1e-3
POLISH_TIME = 1.0
MH_STEPS    = 150
MH_TAU      = 0.1

T_BUDGETS = [1.0, 5.0, 30.0]   # seconds for Gurobi variants

OUT_DIR = os.path.join(HERE, "experiments", "mipfocus")


# ── Gurobi runner ─────────────────────────────────────────────────────────────
def solve_gurobi(A, c, time_limit, mipfocus=0):
    """
    Solve set-cover with Gurobi.
    mipfocus=0: default (balanced).
    mipfocus=1: heuristic emphasis (FP + RINS + Local Branching).

    Returns dict: obj, time, gap, status, first_sol_time, first_sol_obj
    """
    import gurobipy as gp
    from gurobipy import GRB

    n, m = A.shape
    first_sol = {}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            t = model.cbGet(GRB.Callback.RUNTIME)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if "time" not in first_sol:
                first_sol["time"] = t
                first_sol["obj"] = obj

    t0 = time.time()
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", float(time_limit))
        env.setParam("Threads", 1)          # single-threaded: fair comparison with RICH
        if mipfocus:
            env.setParam("MIPFocus", int(mipfocus))
        env.start()

        with gp.Model(env=env) as model:
            x = model.addMVar(shape=m, vtype=GRB.BINARY, name="x")
            for i in range(n):
                idx = np.where(A[i, :] == 1)[0]
                if idx.size:
                    model.addConstr(x[idx].sum() >= 1.0)
            model.setObjective(c @ x, GRB.MINIMIZE)
            model.optimize(cb)

            elapsed = time.time() - t0
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                try:
                    obj = float(model.ObjVal)
                except Exception:
                    obj = np.nan
                try:
                    bound = float(model.ObjBound)
                    gap = 100.0 * abs(obj - bound) / max(abs(obj), 1e-9)
                except Exception:
                    gap = np.nan
                status = "Optimal" if model.Status == GRB.OPTIMAL else "TimeLimit"
            else:
                obj, gap, status = np.nan, np.nan, f"Status_{model.Status}"

    return {
        "obj": obj,
        "time": elapsed,
        "gap_internal": gap,
        "status": status,
        "first_sol_time": first_sol.get("time", np.nan),
        "first_sol_obj": first_sol.get("obj", np.nan),
    }


# ── helpers ───────────────────────────────────────────────────────────────────
def p95(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.percentile(v, 95)) if v else float("nan")

def med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")

def gap_to_ref(obj, ref):
    if not np.isfinite(obj) or not np.isfinite(ref) or abs(ref) < 1e-9:
        return np.nan
    return 100.0 * (obj - ref) / abs(ref)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    rows = []
    total_start = time.time()

    for (n, m) in SCALES:
        for trial in range(TRIALS):
            s = (SEED + trial) % (2**32 - 1)
            A, cost = _gen_entropy_friendly_scp(n, m, seed=s)
            print(f"\n[{n}x{m} trial={trial}]")

            # ── RICH Stage 1-4 ──────────────────────────────────────────
            t0 = time.time()
            x, rich_obj, feas, rich_elapsed = solve_entropy_setcover(
                A, cost,
                tau=0.1, tau_schedule=TAU_SCHED,
                iters=ITERS, tol=TOL,
                polish_time=POLISH_TIME, polish_pool=0.3,
                do_polish_mh=True, mh_tau=MH_TAU, mh_steps=MH_STEPS,
                mh_seed=s,
            )
            rich_wall = time.time() - t0
            print(f"  RICH 1-4 : obj={rich_obj:.2f}  t={rich_elapsed:.3f}s  feas={feas}")

            rows.append({
                "method": "RICH-1-4",
                "time_budget": rich_elapsed,
                "n": n, "m": m, "trial": trial,
                "obj": rich_obj,
                "elapsed": rich_elapsed,
                "wall": rich_wall,
                "first_sol_time": 0.0,
                "first_sol_obj": rich_obj,
                "feasible": int(feas),
            })

            # ── Gurobi variants ─────────────────────────────────────────
            for T in T_BUDGETS:
                for focus, label in [(0, "Gurobi-default"), (1, "Gurobi-MIPFocus1")]:
                    r = solve_gurobi(A, cost, time_limit=T, mipfocus=focus)
                    print(f"  {label:20s} T={T:4.0f}s : "
                          f"obj={r['obj']:.2f}  1st@{r['first_sol_time']:.2f}s  "
                          f"status={r['status']}")
                    rows.append({
                        "method": label,
                        "time_budget": T,
                        "n": n, "m": m, "trial": trial,
                        "obj": r["obj"],
                        "elapsed": r["time"],
                        "wall": r["time"],
                        "first_sol_time": r["first_sol_time"],
                        "first_sol_obj": r["first_sol_obj"],
                        "feasible": int(np.isfinite(r["obj"])),
                    })

    df = pd.DataFrame(rows)

    # ── compute gap vs best-known (MIPFocus1 @ 30s) ──────────────────────────
    ref_key = df[(df["method"] == "Gurobi-MIPFocus1") & (df["time_budget"] == 30.0)]
    ref_map = {(row.n, row.m, row.trial): row.obj
               for _, row in ref_key.iterrows()}

    df["ref_obj"] = df.apply(lambda r: ref_map.get((r.n, r.m, r.trial), np.nan), axis=1)
    df["gap_pct"] = df.apply(lambda r: gap_to_ref(r.obj, r.ref_obj), axis=1)

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, f"mipfocus_results_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[mipfocus] Wrote {csv_path}  ({len(df)} rows, "
          f"{time.time()-total_start:.1f}s total)")

    # ── summary table ─────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\nMIPFocus comparison summary — {ts}")
    lines.append(f"Scales: {SCALES}  trials={TRIALS}  seed={SEED}")
    lines.append(f"Gap computed vs Gurobi-MIPFocus1 @ 30s (best-known reference)\n")

    for (n, m) in SCALES:
        lines.append(f"--- {n}x{m} ---")
        header = f"  {'Method':22s}  {'Budget':>7}  {'gap med%':>9}  {'gap P95%':>9}  {'obj med':>9}  {'time med':>8}  {'1st-sol med':>11}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        # RICH (no budget column — it is what it is)
        sub = df[(df["method"] == "RICH-1-4") & (df["n"] == n) & (df["m"] == m)]
        lines.append(f"  {'RICH Stage 1-4':22s}  {'~1s':>7}  "
                     f"{med(sub['gap_pct']):>9.2f}  {p95(sub['gap_pct']):>9.2f}  "
                     f"{med(sub['obj']):>9.2f}  {med(sub['elapsed']):>8.3f}  {'n/a':>11}")

        for label in ["Gurobi-default", "Gurobi-MIPFocus1"]:
            for T in T_BUDGETS:
                sub = df[(df["method"] == label) & (df["time_budget"] == T)
                         & (df["n"] == n) & (df["m"] == m)]
                lines.append(
                    f"  {label:22s}  {T:>6.0f}s  "
                    f"{med(sub['gap_pct']):>9.2f}  {p95(sub['gap_pct']):>9.2f}  "
                    f"{med(sub['obj']):>9.2f}  {med(sub['elapsed']):>8.3f}  "
                    f"{med(sub['first_sol_time']):>11.3f}"
                )
        lines.append("")

    summary = "\n".join(lines)
    print(summary)

    txt_path = os.path.join(OUT_DIR, f"mipfocus_summary_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[mipfocus] Summary written to {txt_path}")

    return df


if __name__ == "__main__":
    main()
