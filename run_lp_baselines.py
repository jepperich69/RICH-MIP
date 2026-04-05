"""
LP baseline comparison for Sections 3.1 (population synthesis) and 3.2 (transportation).

Both are TU problems, so the LP relaxation is integer-valued — solution quality
is identical between RICH and a direct LP solve. The comparison is purely about speed.

Addresses: TE.10, R2.7.

Methods compared:
  Population synthesis:
    RICH  : IPF relaxation (numpy) + Gurobi residual-LP rounding (Threads=1)
    LP    : Gurobi LP (full transportation feasibility, Threads=1)

  Transportation:
    RICH  : Sinkhorn relaxation + Gurobi LP warm-started from Sinkhorn solution (Threads=1)
    LP    : Gurobi min-cost transport LP cold-start (Threads=1)

Outputs:
  experiments/lp_baselines/lp_baselines_results_<ts>.csv
  experiments/lp_baselines/lp_baselines_summary_<ts>.txt
"""

import sys, os, time, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mip_hybrid.apps.population_transport import (
    make_contingency2d, ipf_2d,
    make_transport, sinkhorn_balanced_uv,
)

OUT_DIR = os.path.join(HERE, "experiments", "lp_baselines")

# ── instance sizes matching the paper (Table 3.1 / 3.2) ─────────────────────
POP_SCALES   = [(80, 80, 80*80*8), (120, 120, 120*120*8), (160, 160, 160*160*8)]
TRANS_SCALES = [(50, 50, 5000), (100, 100, 20000), (200, 200, 80000)]
TRIALS = 5
SEED   = 123


# ── RICH rounding helpers ────────────────────────────────────────────────────

def gurobi_round_population(X_ipf, row_marg, col_marg):
    """
    Residual LP rounding for population synthesis (TU).
    Given fractional X_ipf, solve LP on the floor-residuals to get integer X.
    Variables z_ij in [0,1] with exact row/col deficit constraints.
    TU => LP solution is integral. Threads=1.
    Returns (Xint, time_s).
    """
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.time()
    R, C = X_ipf.shape
    F = np.floor(X_ipf).astype(int)
    dr = (row_marg - F.sum(axis=1)).astype(int)
    dc = (col_marg - F.sum(axis=0)).astype(int)
    frac = X_ipf - F
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        with gp.Model(env=env) as model:
            z_flat = model.addMVar(R * C, lb=0.0, ub=1.0, name="z")
            # row deficit constraints
            for i in range(R):
                idx = np.arange(i * C, (i + 1) * C)
                model.addConstr(z_flat[idx].sum() == float(dr[i]))
            # col deficit constraints
            for j in range(C):
                idx = np.arange(j, R * C, C)
                model.addConstr(z_flat[idx].sum() == float(dc[j]))
            # objective: min sum (1 - frac_ij) * z_ij
            c_obj = (1.0 - frac).flatten()
            model.setObjective(c_obj @ z_flat, GRB.MINIMIZE)
            model.optimize()
            Z = np.round(z_flat.X).astype(int).reshape(R, C)
    Xint = F + Z
    return Xint, time.time() - t0


def gurobi_lp_transport_with_warmstart(supply, demand, C_cost, X_warm=None):
    """
    Solve min-cost transportation LP with Gurobi. TU => integer solution.
    If X_warm is given, use it as primal LP start (PStart). Threads=1.
    Returns (Xint, cost, time_s).
    """
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.time()
    m, n = len(supply), len(demand)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar((m, n), lb=0.0, name="x")
            for i in range(m):
                model.addConstr(x[i, :].sum() == float(supply[i]))
            for j in range(n):
                model.addConstr(x[:, j].sum() == float(demand[j]))
            c_flat = C_cost.flatten()
            x_flat = x.reshape(m * n)
            model.setObjective(c_flat @ x_flat, GRB.MINIMIZE)
            if X_warm is not None:
                # Set LP primal start from warm solution
                x_flat.PStart = X_warm.flatten().astype(float)
            model.optimize()
            X = np.round(x.X).astype(int)
            cost = float((C_cost * x.X).sum())
    return X, cost, time.time() - t0


# ── Gurobi cold LP solvers ───────────────────────────────────────────────────

def gurobi_lp_population(row_marg, col_marg):
    """
    Solve the transportation feasibility LP: find X with given row/col sums.
    TU => LP solution is integer. Threads=1.
    Returns (X, time_s).
    """
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.time()
    R, C = len(row_marg), len(col_marg)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar((R, C), lb=0.0, name="x")
            for i in range(R):
                model.addConstr(x[i, :].sum() == float(row_marg[i]))
            for j in range(C):
                model.addConstr(x[:, j].sum() == float(col_marg[j]))
            model.setObjective(0, GRB.MINIMIZE)
            model.optimize()
            X = np.round(x.X).astype(int)
    return X, time.time() - t0


# ── helpers ──────────────────────────────────────────────────────────────────
def med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")

def p95(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.percentile(v, 95)) if v else float("nan")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    rows = []
    total_start = time.time()

    # ── Population synthesis ─────────────────────────────────────────────────
    print("\n=== Population synthesis: RICH vs Gurobi LP ===")
    for (R, C, N) in POP_SCALES:
        for trial in range(TRIALS):
            s = SEED + trial * 100
            inst = make_contingency2d(R, C, N, seed=s)

            # RICH: IPF (numpy) + Gurobi residual LP rounding
            X_ipf, t_relax = ipf_2d(inst.row_marg, inst.col_marg)
            Xint, t_round  = gurobi_round_population(X_ipf, inst.row_marg, inst.col_marg)
            rich_time = t_relax + t_round
            max_dev   = float(np.abs(Xint - X_ipf).max())

            # Gurobi LP cold (full feasibility LP, no IPF warm-start)
            _, t_lp = gurobi_lp_population(inst.row_marg, inst.col_marg)

            print(f"  [{R}x{C} N={N} t={trial}]  "
                  f"RICH={rich_time:.3f}s (relax={t_relax:.3f}+round={t_round:.3f})  "
                  f"LP={t_lp:.3f}s  max|dev|={max_dev:.3f}")

            rows += [
                {"problem": "population", "R": R, "C": C, "N": N, "trial": trial,
                 "method": "RICH-IPF", "time_s": rich_time,
                 "relax_s": t_relax, "round_s": t_round, "max_dev": max_dev},
                {"problem": "population", "R": R, "C": C, "N": N, "trial": trial,
                 "method": "Gurobi-LP", "time_s": t_lp,
                 "relax_s": np.nan, "round_s": np.nan, "max_dev": 0.0},
            ]

    # ── Transportation ───────────────────────────────────────────────────────
    print("\n=== Transportation: RICH vs Gurobi LP ===")
    for (m, n, N) in TRANS_SCALES:
        for trial in range(TRIALS):
            s = SEED + trial * 100
            inst = make_transport(m, n, N, seed=s)

            # RICH: Sinkhorn + warm-started Gurobi LP
            Xtau, u, v, t_relax = sinkhorn_balanced_uv(
                inst.supply, inst.demand, inst.C, tau=0.05)
            Xint, rich_cost, t_round = gurobi_lp_transport_with_warmstart(
                inst.supply, inst.demand, inst.C, X_warm=Xtau)
            rich_time = t_relax + t_round

            # Gurobi LP cold
            _, lp_cost, t_lp = gurobi_lp_transport_with_warmstart(
                inst.supply, inst.demand, inst.C, X_warm=None)
            gap = 100.0 * abs(rich_cost - lp_cost) / max(abs(lp_cost), 1e-9)

            print(f"  [{m}x{n} N={N} t={trial}]  "
                  f"RICH={rich_time:.3f}s (relax={t_relax:.3f}+round={t_round:.3f})  "
                  f"LP={t_lp:.3f}s  "
                  f"RICH cost={rich_cost:.1f}  LP cost={lp_cost:.1f}  gap={gap:.3f}%")

            rows += [
                {"problem": "transport", "R": m, "C": n, "N": N, "trial": trial,
                 "method": "RICH-Sinkhorn", "time_s": rich_time,
                 "relax_s": t_relax, "round_s": t_round,
                 "obj": rich_cost, "gap_pct": gap},
                {"problem": "transport", "R": m, "C": n, "N": N, "trial": trial,
                 "method": "Gurobi-LP", "time_s": t_lp,
                 "relax_s": np.nan, "round_s": np.nan,
                 "obj": lp_cost, "gap_pct": 0.0},
            ]

    # ── save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"lp_baselines_results_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[lp] Wrote {csv_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\nLP baseline summary -- {ts}")
    lines.append(f"Threads=1 for both RICH rounding and Gurobi-LP; RICH uses IPF/Sinkhorn warm-start\n")

    lines.append("Population synthesis (matching row/col marginals exactly):")
    lines.append(f"  {'Size':>12}  {'RICH med(s)':>12}  {'LP med(s)':>10}  {'Speedup':>8}")
    lines.append("  " + "-" * 50)
    for (R, C, N) in POP_SCALES:
        r = df[(df["problem"]=="population") & (df["R"]==R) & (df["method"]=="RICH-IPF")]
        l = df[(df["problem"]=="population") & (df["R"]==R) & (df["method"]=="Gurobi-LP")]
        rt, lt = med(r["time_s"]), med(l["time_s"])
        sp = lt / rt if rt > 0 else float("nan")
        lines.append(f"  {R}x{C} N={N:>8}  {rt:>12.4f}  {lt:>10.4f}  {sp:>7.1f}x")

    lines.append("\nTransportation (min-cost, gap vs LP = 0% by TU):")
    lines.append(f"  {'Size':>10}  {'RICH med(s)':>12}  {'LP med(s)':>10}  {'Speedup':>8}")
    lines.append("  " + "-" * 48)
    for (m, n, N) in TRANS_SCALES:
        r = df[(df["problem"]=="transport") & (df["R"]==m) & (df["method"]=="RICH-Sinkhorn")]
        l = df[(df["problem"]=="transport") & (df["R"]==m) & (df["method"]=="Gurobi-LP")]
        rt, lt = med(r["time_s"]), med(l["time_s"])
        sp = lt / rt if rt > 0 else float("nan")
        lines.append(f"  {m}x{n} N={N:>6}  {rt:>12.4f}  {lt:>10.4f}  {sp:>7.1f}x")

    summary = "\n".join(lines)
    print(summary)
    txt_path = os.path.join(OUT_DIR, f"lp_baselines_summary_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[lp] Summary written to {txt_path}")
    return df


if __name__ == "__main__":
    main()
