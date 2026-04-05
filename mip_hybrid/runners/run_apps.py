# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:12:25 2025

@author: rich
"""

# -*- coding: utf-8 -*-
"""
RUNNER: Applications benchmarks (Population 2D, Transportation TU)
- Calls apps_population_transport.py
- Writes CSVs
- Summarizes medians/P95
- Prints Overleaf-ready LaTeX tables
"""

import os, datetime as dt, csv
import numpy as np

# Point to your working folder
BASE = r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid"
os.makedirs(BASE, exist_ok=True)

import sys
if BASE not in sys.path:
    sys.path.append(BASE)

import apps_population_transport as ap  # <- your module from earlier

def p95(x): 
    x = np.asarray(list(x), dtype=float)
    return float(np.percentile(x, 95)) if x.size else float("nan")

def write_csv(rows, path):
    if not rows: 
        return None
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows: w.writerow(r)
    return path

def run_population_sweep(scales, trials=3, seed=2025, method="mcf"):
    all_rows = []
    for (R, C, N) in scales:
        rows = ap.bench_population(R=R, C=C, N=N, trials=trials, seed=seed, method=method)
        for r in rows: 
            r.update({"family":"population2d", "R":R, "C":C, "N":N})
        all_rows.extend(rows)
    return all_rows

def run_transport_sweep(scales, trials=3, seed=2025, tau=0.05):
    all_rows = []
    for (m, n, N) in scales:
        rows = ap.bench_transport(m=m, n=n, N=N, trials=trials, seed=seed, tau=tau)
        for r in rows:
            r.update({"family":"transport", "m":m, "n":n, "N":N})
        all_rows.extend(rows)
    return all_rows

def summarize_population(rows):
    # group by (R,C,N)
    out = []
    from collections import defaultdict
    G = defaultdict(list)
    for r in rows:
        G[(r["R"], r["C"], r["N"])].append(r)
    for (R,C,N), rs in sorted(G.items()):
        relax = [r["relax_time"] for r in rs]
        roundt = [r["round_time"] for r in rs]
        maxdev = [r["max_cell_dev"] for r in rs]
        out.append({
            "R":R, "C":C, "N":N, "count":len(rs),
            "relax_med": float(np.median(relax)), "relax_p95": p95(relax),
            "round_med": float(np.median(roundt)), "round_p95": p95(roundt),
            "maxdev_med": float(np.median(maxdev)), "maxdev_p95": p95(maxdev),
        })
    return out

def summarize_transport(rows):
    out = []
    from collections import defaultdict
    G = defaultdict(list)
    for r in rows:
        G[(r["m"], r["n"], r["N"])].append(r)
    for (m,n,N), rs in sorted(G.items()):
        gaps = [r["hyb_gap_pct"] for r in rs]
        optt = [r["opt_time"] for r in rs]
        rela = [r["relax_time"] for r in rs]
        rnd  = [r["round_time"] for r in rs]
        hybt = [r["relax_time"] + r["round_time"] for r in rs]
        out.append({
            "m":m, "n":n, "N":N, "count":len(rs),
            "gap_med": float(np.median(gaps)), "gap_p95": p95(gaps),
            "opt_med": float(np.median(optt)), "opt_p95": p95(optt),
            "relax_med": float(np.median(rela)), "relax_p95": p95(rela),
            "round_med": float(np.median(rnd)), "round_p95": p95(rnd),
            "hyb_med": float(np.median(hybt)), "hyb_p95": p95(hybt),
        })
    return out

def latex_population_table(summary):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{rrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$R{\times}C$ & $N$ & Relax (med) & Relax (P95) & Round (med) & Round (P95) & Max $|\Delta|$ (med) & Max $|\Delta|$ (P95) \\")
    lines.append(r"\midrule")
    for s in summary:
        size = f"{s['R']}\\times{s['C']}"
        lines.append(f"{size} & {s['N']} & {s['relax_med']:.3f} & {s['relax_p95']:.3f} & "
                     f"{s['round_med']:.3f} & {s['round_p95']:.3f} & "
                     f"{s['maxdev_med']:.3f} & {s['maxdev_p95']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Population synthesis (2D). IPF relaxation with exact residual rounding (per-cell deviation $<1$). Times in seconds.}")
    lines.append(r"\label{tab:apps-pop2d}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def latex_transport_table(summary):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{rrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$m{\times}n$ & $N$ & Gap (med,\%) & Gap (P95,\%) & "
                 r"Opt (med) & Opt (P95) & Relax (med) & Round (med) & Hybrid (med) & Hybrid (P95) \\")
    lines.append(r"\midrule")
    for s in summary:
        size = f"{s['m']}\\times{s['n']}"
        lines.append(f"{size} & {s['N']} & {s['gap_med']:.2f} & {s['gap_p95']:.2f} & "
                     f"{s['opt_med']:.3f} & {s['opt_p95']:.3f} & "
                     f"{s['relax_med']:.3f} & {s['round_med']:.3f} & "
                     f"{s['hyb_med']:.3f} & {s['hyb_p95']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Transportation (TU). Entropic relaxation (Sinkhorn) and cost-aware LP integerization (TU $\Rightarrow$ integral). Gap is w.r.t.\ optimal min-cost flow; times in seconds.}")
    lines.append(r"\label{tab:apps-transport}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

if __name__ == "__main__":
    # ----- Define size sweeps -----
    # Population: pick N ≈ 8*R*C (so each cell has decent mass)
    pop_scales = [
        (80, 80, 80*80*8),
        (120, 120, 120*120*8),
        (160, 160, 160*160*8),
    ]
    # Transport: scale total flow with size as well
    tr_scales = [
        (50, 50, 5000),
        (100, 100, 20000),
        (200, 200, 80000),
    ]

    TRIALS = 5
    SEED   = 2025

    # ----- Run benchmarks -----
    pop_rows = run_population_sweep(pop_scales, trials=TRIALS, seed=SEED, method="mcf")
    tr_rows  = run_transport_sweep(tr_scales,  trials=TRIALS, seed=SEED, tau=0.05)

    # ----- Write CSVs -----
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    pop_csv = os.path.join(BASE, f"apps_population2d_{ts}.csv")
    tr_csv  = os.path.join(BASE, f"apps_transport_{ts}.csv")
    write_csv(pop_rows, pop_csv)
    write_csv(tr_rows,  tr_csv)
    print("Wrote:", pop_csv)
    print("Wrote:", tr_csv)

    # ----- Summaries -----
    pop_sum = summarize_population(pop_rows)
    tr_sum  = summarize_transport(tr_rows)

    # ----- LaTeX tables (printed to console) -----
    print("\n\n===== LaTeX: Population Table =====")
    print(latex_population_table(pop_sum))

    print("\n\n===== LaTeX: Transport Table =====")
    print(latex_transport_table(tr_sum))

    # ----- Optionally, save LaTeX to files -----
    with open(os.path.join(BASE, f"apps_population2d_table_{ts}.tex"), "w", encoding="utf-8") as f:
        f.write(latex_population_table(pop_sum))
    with open(os.path.join(BASE, f"apps_transport_table_{ts}.tex"), "w", encoding="utf-8") as f:
        f.write(latex_transport_table(tr_sum))
    print("\nSaved LaTeX tables to BASE.")
