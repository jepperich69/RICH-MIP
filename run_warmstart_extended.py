"""
Extended warm-start analysis for set-cover (Sections 3.3.3 / R2.8 / R2.9).

Addresses reviewer requests:
  R2.8: track dual bounds, node counts, optimality gap (not just primal obj)
  R2.9: extended 120s budget; solve-to-optimality on small instances

Compares:
  COLD : Gurobi cold-start (no initial solution), 120s, Threads=1
  WARM : Gurobi warm-started from RICH Stage 1-4, 120s, Threads=1

Metrics tracked per run:
  - Final primal objective
  - Final dual bound (ObjBound)
  - Optimality gap (%) at end of budget
  - Node count explored
  - Time to first incumbent (cold only)
  - Whether warm start improves on RICH's initial solution

Outputs:
  experiments/warmstart_extended/warmstart_ext_results_<ts>.csv
  experiments/warmstart_extended/warmstart_ext_summary_<ts>.txt
"""

import sys, os, time, datetime
from pathlib import Path
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLEAF_DIR = Path(HERE) / "paper_artifacts"
sys.path.insert(0, HERE)

from mip_hybrid.apps.synth_setcover import (
    _gen_entropy_friendly_scp, solve_entropy_setcover
)

OUT_DIR = os.path.join(HERE, "experiments", "warmstart_extended")

# ── config ────────────────────────────────────────────────────────────────────
# Small scale: 30 s budget (Gurobi typically reaches near-optimality here)
SMALL_SCALES = [(100, 3000)]
# Large scales: 120 s budget (Gurobi cannot close the gap; shows primal improvement)
LARGE_SCALES = [(200, 4000), (300, 5000)]

TRIALS     = 5
SEED       = 42
TIME_LIMIT = 120.0   # seconds for large instances
OPT_LIMIT  =  30.0   # seconds for small instance

TAU_SCHED  = "0.5,0.2,0.1"
ITERS, TOL = 50, 1e-3
POLISH_TIME, MH_STEPS, MH_TAU = 1.0, 150, 0.1


# ── Gurobi runner with full metrics ──────────────────────────────────────────
def solve_gurobi_tracked(A, c, time_limit, x_warm=None):
    """
    Solve set-cover with Gurobi, tracking dual bound, node count, and gap.
    x_warm: if provided, used as MIP start (warm start).
    Uses addVars (dict-based) to support Start attribute.
    Threads=1.

    Returns dict with:
      obj, bound, gap_pct, nodes, time,
      first_sol_time, first_sol_obj, status,
      improved_warm (bool, only meaningful when x_warm given)
    """
    import gurobipy as gp
    from gurobipy import GRB

    n, m = A.shape
    warm_obj = float(c @ x_warm) if x_warm is not None else np.inf
    first_sol = {}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            t   = model.cbGet(GRB.Callback.RUNTIME)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if "time" not in first_sol:
                first_sol["time"] = t
                first_sol["obj"]  = obj

    t0 = time.time()
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit",  float(time_limit))
        env.setParam("Threads",    1)
        env.start()
        with gp.Model(env=env) as model:
            # Use addVars (returns dict) so .Start is accessible on each var
            x = model.addVars(m, vtype=GRB.BINARY, name="x")
            for i in range(n):
                idx = np.where(A[i, :] == 1)[0]
                if idx.size:
                    model.addConstr(
                        gp.quicksum(x[int(j)] for j in idx) >= 1.0
                    )
            model.setObjective(
                gp.quicksum(float(c[j]) * x[j] for j in range(m)),
                GRB.MINIMIZE
            )
            if x_warm is not None:
                for j in range(m):
                    x[j].Start = int(x_warm[j])

            model.optimize(cb)
            elapsed = time.time() - t0

            status = {GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "TimeLimit",
                      GRB.INFEASIBLE: "Infeasible"}.get(model.Status,
                      f"Status_{model.Status}")
            try:
                obj   = float(model.ObjVal)
            except Exception:
                obj   = np.nan
            try:
                bound = float(model.ObjBound)
            except Exception:
                bound = np.nan
            try:
                gap   = float(model.MIPGap) * 100.0
            except Exception:
                gap   = np.nan
            try:
                nodes = int(model.NodeCount)
            except Exception:
                nodes = -1

    return {
        "obj":            obj,
        "bound":          bound,
        "gap_pct":        gap,
        "nodes":          nodes,
        "time":           elapsed,
        "status":         status,
        "first_sol_time": first_sol.get("time", np.nan),
        "first_sol_obj":  first_sol.get("obj",  np.nan),
        "improved_warm":  bool(np.isfinite(obj) and obj < warm_obj - 1e-6),
    }


# ── helpers ───────────────────────────────────────────────────────────────────
def med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _fmt(val, fmt=".1f", fallback="---"):
    """Format a float; return fallback string if not finite."""
    return format(val, fmt) if np.isfinite(val) else fallback


# ── tex writer ────────────────────────────────────────────────────────────────
def write_table37_tex(df, overleaf_dir=OVERLEAF_DIR):
    """Write Table37.tex (tabular body for tab:hybrid_vs_mip) to paper_artifacts/."""
    overleaf_dir = Path(overleaf_dir)
    overleaf_dir.mkdir(parents=True, exist_ok=True)

    all_scales = SMALL_SCALES + LARGE_SCALES

    lines = [
        r"\begin{tabular}{llrrrrrc}",
        r"\toprule",
        r"Scale & Configuration & Obj & Dual bound & Gap (\%) & Nodes & Time (s) & Improved \\",
    ]

    for (n, m) in all_scales:
        sub = df[(df["n"] == n) & (df["m"] == m)]
        rc  = sub[sub["method"] == "RICH"]
        co  = sub[sub["method"] == "Gurobi-cold"]
        wa  = sub[sub["method"] == "Gurobi-warm"]

        lines.append(r"\midrule")

        # RICH row
        lines.append(
            f"\\multirow{{3}}{{*}}{{${n}\\times{m}$}}"
            f" & RICH (Stage~1--4)"
            f" & {_fmt(med(rc['obj']))}"
            f" & ---"
            f" & ---"
            f" & ---"
            f" & {_fmt(med(rc['time']), '.2f')}"
            f" & --- \\\\"
        )
        # Gurobi cold
        lines.append(
            f"  & Gurobi cold start"
            f" & {_fmt(med(co['obj']))}"
            f" & {_fmt(med(co['bound']))}"
            f" & {_fmt(med(co['gap_pct']))}"
            f" & {_fmt(med(co['nodes']), '.0f')}"
            f" & {_fmt(med(co['time']), '.1f')}"
            f" & --- \\\\"
        )
        # Gurobi warm
        impr_pct = 100.0 * float(wa["improved_warm"].mean())
        lines.append(
            f"  & Gurobi warm start"
            f" & {_fmt(med(wa['obj']))}"
            f" & {_fmt(med(wa['bound']))}"
            f" & {_fmt(med(wa['gap_pct']))}"
            f" & {_fmt(med(wa['nodes']), '.0f')}"
            f" & {_fmt(med(wa['time']), '.1f')}"
            f" & {impr_pct:.0f}\\% \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]

    out = overleaf_dir / "Table37.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[warmstart] Table37.tex → {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    rows = []
    total_start = time.time()

    for scale_set, limit in [(SMALL_SCALES, OPT_LIMIT), (LARGE_SCALES, TIME_LIMIT)]:
        for (n, m) in scale_set:
            for trial in range(TRIALS):
                s = (SEED + trial) % (2**32 - 1)
                A, cost = _gen_entropy_friendly_scp(n, m, seed=s)

                # RICH Stage 1-4 warm start — single call, unpack solution vector
                x_warm, rich_obj, feas, rich_t = solve_entropy_setcover(
                    A, cost, tau=0.1, tau_schedule=TAU_SCHED,
                    iters=ITERS, tol=TOL,
                    polish_time=POLISH_TIME, polish_pool=0.3,
                    do_polish_mh=True, mh_tau=MH_TAU, mh_steps=MH_STEPS,
                    mh_seed=s,
                )

                # Cold start
                cold = solve_gurobi_tracked(A, cost, time_limit=limit, x_warm=None)

                # Warm start
                warm = solve_gurobi_tracked(A, cost, time_limit=limit, x_warm=x_warm)

                print(f"  [{n}x{m} t={trial} lim={limit:.0f}s]")
                print(f"    RICH:  obj={rich_obj:.2f}  t={rich_t:.3f}s  feas={feas}")
                print(f"    COLD:  obj={cold['obj']:.2f}  bound={cold['bound']:.2f}  "
                      f"gap={cold['gap_pct']:.2f}%  nodes={cold['nodes']}  "
                      f"status={cold['status']}")
                print(f"    WARM:  obj={warm['obj']:.2f}  bound={warm['bound']:.2f}  "
                      f"gap={warm['gap_pct']:.2f}%  nodes={warm['nodes']}  "
                      f"improved={warm['improved_warm']}")

                base = {"n": n, "m": m, "trial": trial, "time_limit": limit}
                rows.append({**base, "method": "RICH",
                             "obj": rich_obj, "bound": np.nan, "gap_pct": np.nan,
                             "nodes": 0, "time": rich_t, "status": "Heuristic",
                             "first_sol_time": 0.0, "first_sol_obj": rich_obj,
                             "improved_warm": False})
                rows.append({**base, "method": "Gurobi-cold",
                             **{k: cold[k] for k in
                                ["obj","bound","gap_pct","nodes","time",
                                 "status","first_sol_time","first_sol_obj"]},
                             "improved_warm": False})
                rows.append({**base, "method": "Gurobi-warm",
                             **{k: warm[k] for k in
                                ["obj","bound","gap_pct","nodes","time",
                                 "status","first_sol_time","first_sol_obj"]},
                             "improved_warm": warm["improved_warm"]})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"warmstart_ext_results_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[warmstart] Wrote {csv_path}  ({len(df)} rows, "
          f"{time.time()-total_start:.1f}s total)")

    # ── summary ──────────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\nExtended warm-start summary -- {ts}")
    lines.append(f"Threads=1, small={OPT_LIMIT:.0f}s, large={TIME_LIMIT:.0f}s\n")

    for (n, m) in SMALL_SCALES + LARGE_SCALES:
        lim = OPT_LIMIT if (n, m) in SMALL_SCALES else TIME_LIMIT
        sub = df[(df["n"] == n) & (df["m"] == m)]
        rc  = sub[sub["method"] == "RICH"]
        co  = sub[sub["method"] == "Gurobi-cold"]
        wa  = sub[sub["method"] == "Gurobi-warm"]
        lines.append(f"  {n}x{m}  (budget={lim:.0f}s)")
        lines.append(f"    RICH         obj={med(rc['obj']):>8.2f}  t={med(rc['time']):>7.3f}s")
        lines.append(f"    Gurobi-cold  obj={med(co['obj']):>8.2f}  "
                     f"bound={med(co['bound']):>8.2f}  "
                     f"gap={med(co['gap_pct']):>6.2f}%  "
                     f"nodes={med(co['nodes']):>8.0f}")
        lines.append(f"    Gurobi-warm  obj={med(wa['obj']):>8.2f}  "
                     f"bound={med(wa['bound']):>8.2f}  "
                     f"gap={med(wa['gap_pct']):>6.2f}%  "
                     f"nodes={med(wa['nodes']):>8.0f}  "
                     f"improved={wa['improved_warm'].mean():.0%}")
        lines.append("")

    summary = "\n".join(lines)
    print(summary)
    txt_path = os.path.join(OUT_DIR, f"warmstart_ext_summary_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[warmstart] Summary written to {txt_path}")
    write_table37_tex(df)
    return df


if __name__ == "__main__":
    main()
