"""
Ablation study: Stage 1-2 vs 1-3 vs 1-4 for synthetic set-cover.

Stages:
  1-2 : IPF relaxation + dual-guided rounding + drop-fix pruning
          (polish_time=0, do_polish_mh=False)
  1-3 : + Gurobi restricted-MIP polish
          (polish_time=POLISH_TIME, do_polish_mh=False)
  1-4 : + Stage 4 MH polish
          (polish_time=POLISH_TIME, do_polish_mh=True)

Outputs:
  ablation_results_<timestamp>.csv   — raw per-trial rows
  ablation_summary_<timestamp>.txt   — median/P95 table (console + file)

Usage (from repo root, with Gurobi env active):
  python run_ablation_setcover.py
"""

import sys, os, time, datetime
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mip_hybrid.apps.synth_setcover import (
    _gen_entropy_friendly_scp,
    solve_entropy_setcover,
    solve_mip,
)

# ── experiment config ─────────────────────────────────────────────────────────
SCALES     = [(100, 500), (200, 1000), (400, 2000)]
TRIALS     = 5
SEED       = 42

TAU        = 0.1
TAU_SCHED  = "0.5,0.2,0.1"   # same schedule used in paper experiments
ITERS      = 50
TOL        = 1e-3

POLISH_TIME = 1.0   # seconds for Stage 3b Gurobi restricted-MIP polish
POLISH_POOL = 0.3
MH_TAU      = 0.1   # Stage 4 temperature (matches final annealing tau)
MH_STEPS    = 150   # steps per trial

WITH_MIP    = True  # also solve exact MIP for reference gap
MIP_LIMIT   = 30.0  # seconds

OUT_DIR     = os.path.join(HERE, "experiments", "ablation")

STAGES = [
    dict(label="1-2", polish_time=0.0,        do_polish_mh=False),
    dict(label="1-3", polish_time=POLISH_TIME, do_polish_mh=False),
    dict(label="1-4", polish_time=POLISH_TIME, do_polish_mh=True),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def p95(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.percentile(v, 95)) if v else float("nan")

def med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")

# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    rows = []
    total_start = time.time()

    for (n, m) in SCALES:
        for trial in range(TRIALS):
            s = (SEED + trial) % (2**32 - 1)
            A, cost = _gen_entropy_friendly_scp(n, m, seed=s)

            # Exact MIP reference (once per instance)
            mip_obj = np.nan
            if WITH_MIP:
                r = solve_mip(A, cost, timelimit_s=MIP_LIMIT,
                              gurobi_time_limit=MIP_LIMIT,
                              track_gurobi_anytime=False)
                mip_obj = r.get("obj", np.nan) if isinstance(r, dict) else r[0]

            for stage in STAGES:
                t0 = time.time()
                x, obj, feas, elapsed = solve_entropy_setcover(
                    A, cost,
                    tau=TAU,
                    tau_schedule=TAU_SCHED,
                    iters=ITERS, tol=TOL,
                    polish_time=stage["polish_time"],
                    polish_pool=POLISH_POOL,
                    do_polish_mh=stage["do_polish_mh"],
                    mh_tau=MH_TAU,
                    mh_steps=MH_STEPS,
                    mh_seed=s,
                )
                wall = time.time() - t0

                gap_pct = (
                    100.0 * (obj - mip_obj) / max(abs(mip_obj), 1e-9)
                    if np.isfinite(mip_obj) and np.isfinite(obj)
                    else np.nan
                )

                rows.append({
                    "stage": stage["label"],
                    "n": n, "m": m, "trial": trial,
                    "obj": obj,
                    "mip_obj": mip_obj,
                    "gap_pct": gap_pct,
                    "feasible": int(feas),
                    "time_s": elapsed,
                    "wall_s": wall,
                })
                flag = "OK" if feas else "INFEAS"
                print(f"  [{n}x{m} t={trial}] stage={stage['label']}  "
                      f"obj={obj:.2f}  gap={gap_pct:+.1f}%  t={elapsed:.3f}s  {flag}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, f"ablation_results_{ts}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n[ablation] Wrote {csv_path}  ({len(df)} rows, "
          f"{time.time()-total_start:.1f}s total)")

    # ── summary table ─────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\nAblation summary — {ts}")
    lines.append(f"Scales: {SCALES}  trials={TRIALS}  seed={SEED}")
    lines.append(f"(n x m notation)")
    lines.append(f"tau_sched={TAU_SCHED}  mh_steps={MH_STEPS}  mh_tau={MH_TAU}\n")

    header = (f"{'Stage':>7}  {'nxm':>12}  "
              f"{'gap med%':>9}  {'gap P95%':>9}  "
              f"{'obj med':>9}  {'time med':>9}  {'feas%':>6}")
    lines.append(header)
    lines.append("-" * len(header))

    for stage_lbl in ["1-2", "1-3", "1-4"]:
        sub = df[df["stage"] == stage_lbl]
        for (n, m) in SCALES:
            g = sub[(sub["n"] == n) & (sub["m"] == m)]
            feas_pct = 100.0 * g["feasible"].mean()
            lines.append(
                f"{stage_lbl:>7}  {n}x{m:>8}  "
                f"{med(g['gap_pct']):>9.2f}  {p95(g['gap_pct']):>9.2f}  "
                f"{med(g['obj']):>9.2f}  {med(g['time_s']):>9.3f}  "
                f"{feas_pct:>6.0f}%"
            )
        lines.append("")

    summary = "\n".join(lines)
    print(summary)

    txt_path = os.path.join(OUT_DIR, f"ablation_summary_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[ablation] Summary written to {txt_path}")

    return df


if __name__ == "__main__":
    main()
