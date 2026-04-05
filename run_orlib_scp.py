"""
OR-Library SCP benchmark runner.
Tests RICH Stage 1-4 and Gurobi (default, MIPFocus=1) on the standard
scp4x series (200x1000, ~2% density).

Known optima (Beasley 1987/1990, widely cited):
  scp41=429, scp42=512, scp43=516, scp44=494, scp45=512,
  scp46=560, scp47=430, scp48=492, scp49=641

Addresses reviewer comments R1.5 / R3.7 (standard benchmark instances).

Usage (from repo root, Gurobi env active):
  python run_orlib_scp.py
"""

import sys, os, time, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mip_hybrid.apps.synth_setcover import solve_entropy_setcover

# ── known optima (Beasley 1987) ───────────────────────────────────────────────
KNOWN_OPT = {
    "scp41": 429, "scp42": 512, "scp43": 516,
    "scp44": 494, "scp45": 512, "scp46": 560,
    "scp47": 430, "scp48": 492, "scp49": 641,
}

DATA_DIR = os.path.join(HERE, "data", "orlib_scp")
OUT_DIR  = os.path.join(HERE, "experiments", "orlib_scp")

# ── RICH config (matches ablation study) ─────────────────────────────────────
TAU_SCHED   = "0.5,0.2,0.1"
ITERS       = 50
TOL         = 1e-3
POLISH_TIME = 1.0
MH_STEPS    = 150
MH_TAU      = 0.1

MIP_TIME    = 30.0   # Gurobi reference budget (single thread)


# ── OR-Library SCP parser ─────────────────────────────────────────────────────
def load_orlib_scp(path):
    """
    Parse standard OR-Library SCP format (row-oriented, Beasley).

    Format:
      nrows ncols
      c_1 c_2 ... c_ncols          (column costs, split across lines)
      for each row i (1..nrows):
        n_i  col_1 col_2 ... col_{n_i}   (1-indexed column indices)

    Returns:
      A  : numpy bool matrix (nrows x ncols)
      c  : numpy float vector (ncols,)
    """
    with open(path) as f:
        tokens = f.read().split()

    it = iter(tokens)
    nrows = int(next(it))
    ncols = int(next(it))

    c = np.array([float(next(it)) for _ in range(ncols)])

    A = np.zeros((nrows, ncols), dtype=np.uint8)
    for i in range(nrows):
        n_cols_covering = int(next(it))
        for _ in range(n_cols_covering):
            j = int(next(it)) - 1    # convert 1-indexed -> 0-indexed
            A[i, j] = 1

    return A, c


# ── Gurobi runner (single-threaded) ──────────────────────────────────────────
def solve_gurobi(A, c, time_limit, mipfocus=0):
    import gurobipy as gp
    from gurobipy import GRB

    n, m = A.shape
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
                    gap_int = 100.0 * abs(obj - bound) / max(abs(obj), 1e-9)
                except Exception:
                    gap_int = np.nan
                status = "Optimal" if model.Status == GRB.OPTIMAL else "TimeLimit"
            else:
                obj, gap_int, status = np.nan, np.nan, f"Status_{model.Status}"

    return {
        "obj": obj, "time": elapsed,
        "gap_internal": gap_int, "status": status,
        "first_sol_time": first_sol.get("time", np.nan),
        "first_sol_obj":  first_sol.get("obj",  np.nan),
    }


# ── helpers ───────────────────────────────────────────────────────────────────
def gap_pct(obj, opt):
    if not np.isfinite(obj) or not np.isfinite(opt) or abs(opt) < 1e-9:
        return np.nan
    return 100.0 * (obj - opt) / abs(opt)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    instances = sorted(KNOWN_OPT.keys())   # scp41 .. scp49
    rows = []
    total_start = time.time()

    for name in instances:
        path = os.path.join(DATA_DIR, f"{name}.txt")
        if not os.path.isfile(path):
            print(f"[SKIP] {name} — file not found: {path}")
            continue

        A, cost = load_orlib_scp(path)
        opt = KNOWN_OPT[name]
        n, m = A.shape
        density = A.sum() / (n * m)
        print(f"\n[{name}]  {n}x{m}  density={density:.3f}  opt={opt}")

        # ── RICH Stage 1-4 ──────────────────────────────────────────────────
        t0 = time.time()
        x, rich_obj, feas, rich_elapsed = solve_entropy_setcover(
            A, cost,
            tau=0.1, tau_schedule=TAU_SCHED,
            iters=ITERS, tol=TOL,
            polish_time=POLISH_TIME, polish_pool=0.3,
            do_polish_mh=True, mh_tau=MH_TAU, mh_steps=MH_STEPS,
            mh_seed=0,
        )
        rich_wall = time.time() - t0
        rich_gap = gap_pct(rich_obj, opt)
        print(f"  RICH 1-4      : obj={rich_obj:.0f}  gap={rich_gap:+.2f}%  "
              f"t={rich_elapsed:.3f}s  feas={feas}")

        rows.append({
            "instance": name, "nrows": n, "ncols": m, "density": density,
            "known_opt": opt,
            "method": "RICH-1-4",
            "obj": rich_obj, "gap_pct": rich_gap,
            "elapsed": rich_elapsed, "wall": rich_wall,
            "feasible": int(feas),
            "first_sol_time": 0.0, "first_sol_obj": rich_obj,
        })

        # ── Gurobi default @ 30s ────────────────────────────────────────────
        r = solve_gurobi(A, cost, time_limit=MIP_TIME, mipfocus=0)
        g_gap = gap_pct(r["obj"], opt)
        print(f"  Gurobi-default: obj={r['obj']:.0f}  gap={g_gap:+.2f}%  "
              f"t={r['time']:.1f}s  status={r['status']}")
        rows.append({
            "instance": name, "nrows": n, "ncols": m, "density": density,
            "known_opt": opt,
            "method": "Gurobi-default-30s",
            "obj": r["obj"], "gap_pct": g_gap,
            "elapsed": r["time"], "wall": r["time"],
            "feasible": int(np.isfinite(r["obj"])),
            "first_sol_time": r["first_sol_time"],
            "first_sol_obj":  r["first_sol_obj"],
        })

        # ── Gurobi MIPFocus=1 @ 30s ─────────────────────────────────────────
        r1 = solve_gurobi(A, cost, time_limit=MIP_TIME, mipfocus=1)
        g1_gap = gap_pct(r1["obj"], opt)
        print(f"  Gurobi-MF1    : obj={r1['obj']:.0f}  gap={g1_gap:+.2f}%  "
              f"t={r1['time']:.1f}s  status={r1['status']}")
        rows.append({
            "instance": name, "nrows": n, "ncols": m, "density": density,
            "known_opt": opt,
            "method": "Gurobi-MIPFocus1-30s",
            "obj": r1["obj"], "gap_pct": g1_gap,
            "elapsed": r1["time"], "wall": r1["time"],
            "feasible": int(np.isfinite(r1["obj"])),
            "first_sol_time": r1["first_sol_time"],
            "first_sol_obj":  r1["first_sol_obj"],
        })

    # ── save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"orlib_scp_results_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[orlib] Wrote {csv_path}  ({len(df)} rows, "
          f"{time.time()-total_start:.1f}s total)")

    # ── summary table ─────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\nOR-Library SCP results — {ts}")
    lines.append(f"Gap computed vs known optimum (Beasley 1987)")
    lines.append(f"All methods: Threads=1, RICH polish_time={POLISH_TIME}s\n")

    header = (f"  {'Instance':10s}  {'Opt':>5}  "
              f"{'RICH obj':>8}  {'RICH gap%':>9}  {'RICH t':>7}  "
              f"{'GurDef obj':>10}  {'GurDef gap%':>11}  "
              f"{'GurMF1 obj':>10}  {'GurMF1 gap%':>11}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for name in instances:
        sub  = df[df["instance"] == name]
        rich = sub[sub["method"] == "RICH-1-4"].iloc[0]
        gd   = sub[sub["method"] == "Gurobi-default-30s"].iloc[0]
        gm   = sub[sub["method"] == "Gurobi-MIPFocus1-30s"].iloc[0]
        lines.append(
            f"  {name:10s}  {int(KNOWN_OPT[name]):>5}  "
            f"{rich['obj']:>8.0f}  {rich['gap_pct']:>+9.2f}  {rich['elapsed']:>7.3f}  "
            f"{gd['obj']:>10.0f}  {gd['gap_pct']:>+11.2f}  "
            f"{gm['obj']:>10.0f}  {gm['gap_pct']:>+11.2f}"
        )

    # aggregates
    rich_rows = df[df["method"] == "RICH-1-4"]
    gd_rows   = df[df["method"] == "Gurobi-default-30s"]
    gm_rows   = df[df["method"] == "Gurobi-MIPFocus1-30s"]
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(
        f"  {'median':10s}  {'':>5}  "
        f"{float(np.median(rich_rows['obj'])):>8.0f}  "
        f"{float(np.median(rich_rows['gap_pct'])):>+9.2f}  "
        f"{float(np.median(rich_rows['elapsed'])):>7.3f}  "
        f"{float(np.median(gd_rows['obj'])):>10.0f}  "
        f"{float(np.median(gd_rows['gap_pct'])):>+11.2f}  "
        f"{float(np.median(gm_rows['obj'])):>10.0f}  "
        f"{float(np.median(gm_rows['gap_pct'])):>+11.2f}"
    )

    summary = "\n".join(lines)
    print(summary)

    txt_path = os.path.join(OUT_DIR, f"orlib_scp_summary_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[orlib] Summary written to {txt_path}")

    return df


if __name__ == "__main__":
    main()
