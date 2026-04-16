"""
reproduce_paper.py
==================
Master runner that reproduces all experimental results reported in:

  RICH: A Rapid Information-Theoretic Hybrid Algorithm for Mixed-Integer
  Optimization (Mathematical Programming Computation, R1B revision)

Runs the following experiments in order:

  Section 3.1/3.2 -- LP baseline comparisons (Tables tab:popsyn_lp, tab:transport_lp)
    -> run_lp_baselines.py

  Section 3.3.1 -- Anytime trajectories, Figures 1 & 2
    -> run_anytime_comprehensive.py   (also generates figures → Overleaf_source/)

  Section 3.3.2 -- Stage-contribution ablation (Table tab:ablation → Table35.tex)
    -> run_ablation_setcover.py

  Section 3.3.2 -- OR-Library benchmarks scp41-scp49 (Table tab:orlib → Table36.tex)
    -> run_orlib_scp.py

  Section 3.3.2 -- MIPFocus=1 heuristic comparison (Table in response letter)
    -> run_mipfocus_comparison.py

  Section 3.3.3 -- Extended warm-start analysis (Table tab:hybrid_vs_mip → Table37.tex)
    -> run_warmstart_extended.py

Requirements
------------
  - Python 3.9+
  - numpy, pandas, matplotlib  (see requirements.txt)
  - Gurobi 10+ with valid licence (gurobipy must be importable)
  - The OR-Library data files must be present in data/orlib_scp/
    (scp41.txt -- scp49.txt; download from https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html)

Estimated runtime (single-threaded, modern laptop)
---------------------------------------------------
  LP baselines          ~  2 min
  Anytime (20 trials)   ~ 35 min  (also generates Figures 1 & 2)
  Ablation (5 trials)   ~  8 min  (writes Table35.tex → Overleaf_source/)
  OR-Library (9 inst)   ~  1 min  (writes Table36.tex → Overleaf_source/)
  MIPFocus comparison   ~  5 min
  Warm-start extended   ~ 15 min  (writes Table37.tex → Overleaf_source/)
  ---------------------------------
  Total                 ~ 65 min

Usage
-----
  python reproduce_paper.py              # run all experiments
  python reproduce_paper.py --skip warmstart   # skip one section
  python reproduce_paper.py --only ablation    # run one section only

Output
------
  CSV / TXT results  → experiments/<name>/
  LaTeX table bodies → Overleaf_source/Table35.tex, Table36.tex, Table37.tex
  Figures            → Overleaf_source/time_quality_figure.{png,svg}
                       Overleaf_source/time_advantage_figure.{png,svg}
"""

import argparse
import importlib
import sys
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXPERIMENTS = [
    {
        "key":    "lp_baselines",
        "module": "run_lp_baselines",
        "label":  "§3.1/3.2  LP baseline comparison (Tables tab:popsyn_lp, tab:transport_lp)",
    },
    {
        "key":    "anytime",
        "module": "run_anytime_comprehensive",
        "label":  "§3.3.1    Anytime trajectories + Figures 1 & 2 (→ Overleaf_source/)",
    },
    {
        "key":    "ablation",
        "module": "run_ablation_setcover",
        "label":  "§3.3.2    Stage-contribution ablation (Table35.tex → Overleaf_source/)",
    },
    {
        "key":    "orlib",
        "module": "run_orlib_scp",
        "label":  "§3.3.2    OR-Library benchmarks scp41-scp49 (Table36.tex → Overleaf_source/)",
    },
    {
        "key":    "mipfocus",
        "module": "run_mipfocus_comparison",
        "label":  "§3.3.2    Gurobi MIPFocus=1 heuristic comparison",
    },
    {
        "key":    "warmstart",
        "module": "run_warmstart_extended",
        "label":  "§3.3.3    Extended warm-start analysis (Table37.tex → Overleaf_source/)",
    },
]


def check_gurobi():
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
        print("  Gurobi: OK")
        return True
    except Exception as e:
        print(f"  Gurobi: FAILED ({e})")
        return False


def check_orlib_data():
    data_dir = os.path.join(HERE, "data", "orlib_scp")
    missing = []
    for i in range(1, 10):
        fname = os.path.join(data_dir, f"scp4{i}.txt")
        if not os.path.exists(fname):
            missing.append(f"scp4{i}.txt")
    if missing:
        print(f"  OR-Library data: MISSING {missing}")
        print(f"    Download from https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html")
        print(f"    and place in {data_dir}/")
        return False
    print("  OR-Library data: OK")
    return True


def run_experiment(exp):
    print(f"\n{'='*70}")
    print(f"Running: {exp['label']}")
    print(f"{'='*70}")
    t0 = time.time()
    mod = importlib.import_module(exp["module"])
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper experiments.")
    parser.add_argument("--skip", nargs="+", metavar="KEY",
                        help="Skip experiment(s) by key (lp_baselines, anytime, ablation, orlib, mipfocus, warmstart)")
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="Run only the specified experiment(s)")
    args = parser.parse_args()

    print("\nRICH — Paper Reproduction Script")
    print("=" * 70)

    # Environment checks
    print("\nEnvironment checks:")
    gurobi_ok = check_gurobi()
    orlib_ok  = check_orlib_data()

    if not gurobi_ok:
        print("\nERROR: Gurobi is required. Aborting.")
        sys.exit(1)

    # Determine which experiments to run
    to_run = EXPERIMENTS
    if args.only:
        to_run = [e for e in EXPERIMENTS if e["key"] in args.only]
    if args.skip:
        to_run = [e for e in to_run if e["key"] not in args.skip]

    if not orlib_ok and any(e["key"] == "orlib" for e in to_run):
        print("\nWARNING: OR-Library data missing — skipping orlib experiment.")
        to_run = [e for e in to_run if e["key"] != "orlib"]

    print(f"\nWill run {len(to_run)} experiment(s):")
    for e in to_run:
        print(f"  [{e['key']}]  {e['label']}")

    # Run
    total_start = time.time()
    results = {}
    for exp in to_run:
        try:
            elapsed = run_experiment(exp)
            results[exp["key"]] = ("OK", elapsed)
        except Exception as ex:
            print(f"\nERROR in {exp['key']}: {ex}")
            results[exp["key"]] = ("FAILED", 0.0)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for exp in to_run:
        status, t = results.get(exp["key"], ("NOT RUN", 0))
        mark = "OK" if status == "OK" else "FAILED"
        print(f"  [{mark:6}] {exp['key']:20s}  {t:6.1f}s  {exp['label']}")

    total = time.time() - total_start
    print(f"\nTotal wall time: {total:.1f}s")
    print("\nOutput files written to experiments/<name>/")
    print("Done.")


if __name__ == "__main__":
    main()
