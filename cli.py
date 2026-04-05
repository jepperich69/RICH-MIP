# --- bootstrap: make package importable inside Spyder ---
import os, sys
import argparse            
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid\MIP Hybrid Solver")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
# --- end bootstrap ---



# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 18:14:38 2025

@author: rich
"""

print("CWD:", os.getcwd())
print("sys.path[0:3]:", sys.path[0:3])
print("Root exists?", PROJECT_ROOT.exists())
print("Root entries:", [p.name for p in PROJECT_ROOT.iterdir()][:10])
print("Has src/mip_hybrid?", (PROJECT_ROOT/"src"/"mip_hybrid").exists())
print("Has mip_hybrid at root?", (PROJECT_ROOT/"mip_hybrid").exists())


from pathlib import Path

p = PROJECT_ROOT / "mip_hybrid"
print("mip_hybrid is_dir:", p.is_dir())
print("mip_hybrid entries:", [q.name for q in p.iterdir()])

apps = p / "apps"
print("apps exists:", apps.exists(), "is_dir:", apps.is_dir())
print("pkg __init__.py exists:", (p / "__init__.py").exists())
print("apps __init__.py exists:", (apps / "__init__.py").exists())

print("synth_setcover.py exists:", (apps / "synth_setcover.py").exists())
# If you used a different file name, check it here:
# print("synth_set_cover.py exists:", (apps / "synth_set_cover.py").exists())


###



RAIL582_FALLBACK = r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid\rail582"

# ------------------------------
# Subcommand implementations
# ------------------------------

# Reuse your existing RAIL582_FALLBACK; if it's not defined yet, define it here.
# RAIL582_FALLBACK = r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid\rail582"

def _resolve_rail_dir(arg_rail_dir: str | None) -> str:
    cand = arg_rail_dir or os.environ.get("RAIL582_DIR") or RAIL582_FALLBACK
    if not cand:
        raise SystemExit("No RAIL582 directory provided. Use --rail_dir or set RAIL582_DIR.")
    return cand

def _resolve_out_dir(arg_out_dir: str | None, default_rel: str = "experiments/rail") -> str:
    out_dir = arg_out_dir or default_rel
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def run_apps_cmd(args: argparse.Namespace):
    """Legacy population/transport runner (optional)."""
    try:
        from mip_hybrid.runners import run_apps as _legacy_runner
    except ImportError:
        print("[apps] legacy runner not found (mip_hybrid.runners.run_apps). Skipping.")
        return
    if hasattr(_legacy_runner, "main"):
        _legacy_runner.main()
    else:
        __import__("mip_hybrid.runners.run_apps")

def run_rail_cmd(args: argparse.Namespace):
    """
    Run RAIL + optional synthetic via CLI, with polish params controlled here.
    """
    import os, datetime as dt

    # optional synthetic pre-run (keep if you want)
    syn_list = None
    if args.quick and not args.syn:
        syn_list = [(2000, 40000, 1, 0.0, 412)]
    elif args.syn:
        syn_list = []
        for token in args.syn.split(","):
            try:
                n_str, m_str = token.split(":")
                syn_list.append((int(n_str), int(m_str), 1, 0.0, 42))
            except ValueError:
                raise SystemExit(f"--syn must look like 'n:m[,n:m...]'; bad token: {token!r}")

    # Ensure out_dir exists
    out_dir = _resolve_out_dir(getattr(args, "out_dir", None))

    # If running synthetic via the runner, forward polish knobs for consistency
    if syn_list:
        from mip_hybrid.runners.run_rail import run_rail
        res_syn = run_rail(
            args.rail_dir,
            syn_list,
            out_dir,
            trials=args.trials,
            seed=args.seed,
            solver=args.solver,
            polish_time=args.polish_time,
            polish_pool=args.polish_pool,
            rr_trials=args.rr_trials,
            threads=args.threads, 
            mip_gap=args.mip_gap,
            # Add missing Gurobi parameters
            gurobi_time_limit=args.gurobi_time_limit,
            gurobi_gap_limit=args.gurobi_gap_limit,
            track_gurobi_anytime=args.track_gurobi_anytime
        )
        if isinstance(res_syn, dict):
            print(f"[rail] syn csv: {res_syn.get('csv','')}")
            print(f"[rail] syn log: {res_syn.get('log','')}")
    
    # Only run RAIL582 if NOT doing synthetic (or if explicitly requested)
    # You could add a flag like --include_rail if you want both synthetic AND rail
    if not syn_list:  # Only run RAIL582 if no synthetic requested
        rail_dir = _resolve_rail_dir(args.rail_dir)
        from mip_hybrid.apps import rail_setcover as rail_app

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(out_dir, f"rail582_results_{ts}.csv")

        rail_app.main_from_cli_namespace(
            rail_path=rail_dir,
            out=out_csv,                 # important: app writes exactly here
            out_dir=out_dir,
            trials=args.trials,
            solver=args.solver,
            polish_time=args.polish_time,
            polish_pool=args.polish_pool,
            rr_trials=args.rr_trials,
            threads=args.threads,
            mip_gap=args.mip_gap,
            seed=args.seed,
            # Add missing Gurobi parameters
            gurobi_time_limit=args.gurobi_time_limit,
            gurobi_gap_limit=args.gurobi_gap_limit,
            track_gurobi_anytime=args.track_gurobi_anytime
        )
        print(f"[rail] csv: {out_csv}")
    elif syn_list:
        print("[rail] Synthetic run completed. Use 'python cli.py rail' (no --syn) to run RAIL582 separately.")

def run_all_cmd(args: argparse.Namespace):
    """Run the full sweep: synthetic (HYB+MIP) then RAIL582 (HYB+MIP)."""
    from mip_hybrid.runners.run_rail import run_rail
    out_dir = "experiments/rail"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Synthetic sweep (HYB + MIP)
    syn_list = [(200, 4000, 1, 0.0, 42),
                (400, 8000, 1, 0.0, 7),
                (800, 16000, 1, 0.0, 1),
                (1600, 32000, 1, 0.0, 11),                
                ]
#    syn_list = [(2000, 40000, 1, 0.0, 42),
#            (5000, 100000, 1, 0.0, 7)]
    print(f"[DEBUG] About to call run_rail with track_gurobi_anytime=True")
    print("\n[all] Synthetic sweep (HYB + MIP)")
    res_syn = run_rail(
        None,
        syn_list,
        out_dir,
        trials=args.trials,
        seed=args.seed,
        solver=args.solver,
        improve_passes=args.improve_passes,
        threads=args.threads,
        mip_gap=args.mip_gap,
        gurobi_time_limit=args.gurobi_time_limit,
        gurobi_gap_limit=args.gurobi_gap_limit,
        track_gurobi_anytime=True,
#        track_gurobi_anytime=args.track_gurobi_anytime,
    )
    if isinstance(res_syn, dict):
        print(f"[all] synth csv: {res_syn.get('csv','')}")
        print(f"[all] synth stats csv: {res_syn.get('csv_stats','')}")
        print(f"[all] synth tex (story): {res_syn.get('tex_story','')}")
        print(f"[all] synth tex (full):  {res_syn.get('tex_full','')}")
        print(f"[all] synth tex (detailed):  {res_syn.get('tex_detailed','')}")
        print(f"[all] synth log: {res_syn.get('log','')}")

    # 2) RAIL 582 (HYB + MIP)
    rail_dir = _resolve_rail_dir(args.rail_dir)
    out_dir  = _resolve_out_dir(getattr(args, "out_dir", None))
#    rail_dir = args.rail_dir or os.getenv("RAIL582_DIR", "").strip()
    if not rail_dir:
        if os.path.isdir(RAIL582_FALLBACK):
            rail_dir = RAIL582_FALLBACK

    def _looks_like_rail_dir(p: str) -> bool:
        if not p or not os.path.isdir(p):
            return False
        try:
            entries = {name.lower() for name in os.listdir(p)}
        except Exception:
            return False
        return ("rail582" in entries) or ("rail582.txt" in entries)

    print(f"\n[all] RAIL582 path: {rail_dir or '(not set)'}")
    if rail_dir and _looks_like_rail_dir(rail_dir):
        print("[all] RAIL 582 (HYB + MIP)")
        res_rail = run_rail(
            rail_dir, None, out_dir,
            trials=args.trials,
            seed=args.seed,
            solver=args.solver,
            polish_time=args.polish_time,
            polish_pool=args.polish_pool,
            rr_trials=args.rr_trials,
            improve_passes=args.improve_passes, 
            threads=args.threads, 
            mip_gap=args.mip_gap,
            # ADD THESE THREE:
            gurobi_time_limit=args.gurobi_time_limit,
            gurobi_gap_limit=args.gurobi_gap_limit,
            track_gurobi_anytime=args.track_gurobi_anytime
        )        
              
        if isinstance(res_rail, dict):
            print(f"[all] rail csv: {res_rail.get('csv','')}")
            print(f"[all] rail stats csv: {res_rail.get('csv_stats','')}")
            print(f"[all] rail tex (story): {res_rail.get('tex_story','')}")
            print(f"[all] rail tex (full):  {res_rail.get('tex_full','')}")
            print(f"[all] rail tex (detailed):  {res_rail.get('tex_detailed','')}")
            print(f"[all] rail log: {res_rail.get('log','')}")
    else:
        print("[all] RAIL582 skipped: set --rail_dir, env RAIL582_DIR, or update RAIL582_FALLBACK.")

def run_suite_cmd(args: argparse.Namespace):
    """
    Run: (1) population_transport, (2) synthetic set-cover, (3) RAIL582,
    all with identical trials/seed plumbing and the same LaTeX tables.
    """
    from mip_hybrid.runners.run_rail import run_rail, run_one_app
    os.makedirs(args.out_dir, exist_ok=True)

    # (1) population_transport  — adjust module/flags here if your app differs
    pop_out = os.path.join(args.out_dir, "pop_results.csv")
    pop_argv = ["--out", pop_out, "--trials", str(args.trials)]
    if args.seed is not None:
        pop_argv += ["--seed", str(args.seed)]
    res_pop = run_one_app(
        module="mip_hybrid.apps.population_transport",
        argv=pop_argv,
        family="POPTRANS",
        out_dir=args.out_dir,
    )
    if isinstance(res_pop, dict):
        print(f"[suite] pop csv: {res_pop.get('csv','')}")
        print(f"[suite] pop stats csv: {res_pop.get('csv_stats','')}")
        print(f"[suite] pop tex (story): {res_pop.get('tex_story','')}")
        print(f"[suite] pop tex (full):  {res_pop.get('tex_full','')}")
        print(f"[suite] pop tex (detailed):  {res_pop.get('tex_detailed','')}")
        print(f"[suite] pop log: {res_pop.get('log','')}")

    # (2) synthetic set-cover (example small scale)
    res_syn = run_rail(
        None,
        [(200, 4000, 1, 0.0, 142),
                    (300, 6000, 1, 0.0, 17)],        
        args.out_dir,
        trials=args.trials,
        seed=args.seed,
        solver=args.solver,
        improve_passes=args.improve_passes,
        threads=args.threads,
        mip_gap=args.mip_gap,
        gurobi_time_limit=args.gurobi_time_limit,
        gurobi_gap_limit=args.gurobi_gap_limit,
        track_gurobi_anytime=args.track_gurobi_anytime
    )
    if isinstance(res_syn, dict):
        print(f"[suite] syn csv: {res_syn.get('csv','')}")
        print(f"[suite] syn stats csv: {res_syn.get('csv_stats','')}")
        print(f"[suite] syn tex (story): {res_syn.get('tex_story','')}")
        print(f"[suite] syn tex (full):  {res_syn.get('tex_full','')}")
        print(f"[suite] syn tex (detailed):  {res_syn.get('tex_detailed','')}")
        print(f"[suite] syn log: {res_syn.get('log','')}")

    rail_dir = _resolve_rail_dir(args.rail_dir)
    out_dir  = _resolve_out_dir(getattr(args, "out_dir", None))
    # (3) RAIL582
    res_rail = run_rail(
        rail_dir, None, out_dir,
        trials=args.trials,
        seed=args.seed,
        solver=args.solver,
        polish_time=args.polish_time,
        polish_pool=args.polish_pool,
        rr_trials=args.rr_trials,
        improve_passes=args.improve_passes, 
        threads=args.threads, 
        mip_gap=args.mip_gap,
        gurobi_time_limit=args.gurobi_time_limit,      # ← ADD THIS
        gurobi_gap_limit=args.gurobi_gap_limit,        # ← ADD THIS  
        track_gurobi_anytime=args.track_gurobi_anytime # ← ADD THIS
    )    
    if isinstance(res_rail, dict):
        print(f"[suite] rail csv: {res_rail.get('csv','')}")
        print(f"[suite] rail stats csv: {res_rail.get('csv_stats','')}")
        print(f"[suite] rail tex (story): {res_rail.get('tex_story','')}")
        print(f"[suite] rail tex (full):  {res_rail.get('tex_full','')}")
        print(f"[suite] rail tex (detailed):  {res_rail.get('tex_detailed','')}")
        print(f"[suite] rail log: {res_rail.get('log','')}")


def run_pop_cmd(args: argparse.Namespace):
    """Run only population and transport experiments"""
    from mip_hybrid.runners.run_rail import run_one_app
    
    out_dir = _resolve_out_dir(args.out_dir)
    pop_out = os.path.join(out_dir, "pop_results.csv")
    pop_argv = ["--out", pop_out, "--trials", str(args.trials), "--seed", str(args.seed)]
    
    res_pop = run_one_app(
        module="mip_hybrid.apps.population_transport",
        argv=pop_argv,
        family="POPTRANS",
        out_dir=out_dir,
    )
    
    if isinstance(res_pop, dict):
        print(f"[pop] csv: {res_pop.get('csv','')}")

# 2. Add this new command function
def run_warmstart_cmd(args: argparse.Namespace):
    """Run warm-start comparison experiments"""
    import datetime as dt
    
    out_dir = _resolve_out_dir(args.out_dir, default_rel="experiments/warmstart")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"warmstart_results_{ts}.csv")
    
    # Parse scales
    if args.scales:
        scales = []
        for token in args.scales.split(","):
            try:
                n_str, m_str = token.split(":")
                scales.append((int(n_str), int(m_str)))
            except ValueError:
                raise SystemExit(f"--scales must be 'n:m[,n:m...]'; bad token: {token!r}")
    else:
        scales = [(200, 4000), (400, 8000), (800, 16000)]
    
    print(f"[warmstart] Running on scales: {scales}")
    print(f"[warmstart] Trials: {args.trials}")
    print(f"[warmstart] MIP time limit: {args.mip_timelimit}s")
    print(f"[warmstart] Polish: time={args.polish_time}s, pool={args.polish_pool}, rr={args.rr_trials}")  # ADD THIS
    
    df = run_warmstart_comparison(
        scales=scales,
        trials=args.trials,
        out_path=out_csv,
        tau=args.tau,
        seed=args.seed,
        mip_timelimit=args.mip_timelimit,
        tau_schedule=args.tau_schedule,
        iters=args.iters,
        tol=args.tol,
        polish_time=args.polish_time,      # ADD THIS
        polish_pool=args.polish_pool,      # ADD THIS
        rr_trials=args.rr_trials          # ADD THIS
    )
    
    print(f"\n[warmstart] Results saved to: {out_csv}")
    return df

# ------------------------------
# Parser
# ------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="MIP Hybrid Solver CLI")
    sub = p.add_subparsers(dest="app")

    # --- rail subcommand (manual control) ---
    rail = sub.add_parser("rail", help="Run RAIL 582 and/or synthetic set-cover")
    rail.add_argument("--rail_dir", type=str, default=None,
                      help="Path to OR-Library rail582 folder (or set env RAIL582_DIR)")
    rail.add_argument("--out_dir", type=str, default="experiments/rail")
    rail.add_argument("--syn", type=str, default="",
                      help="Comma-separated n:m pairs, e.g. 200:4000,400:8000")
    rail.add_argument("--quick", action="store_true",
                      help="Run a small default synthetic instance if no args given")
    rail.add_argument("--trials", type=int, default=50,
                      help="Number of independent trials to run (default: 10)")
    rail.add_argument("--seed", type=int, default=3221, #4221
                      help="Base RNG seed (apps may offset per trial)")
    rail.add_argument("--solver", choices=["cbc", "gurobi"], default="gurobi",
                      help="MIP solver backend for the RAIL runs (default: cbc).")
    # Hybrid tuning knobs
    rail.add_argument("--polish_time", type=float, default=1.5,
                      help="Hybrid polish time in seconds")
    rail.add_argument("--polish_pool", type=float, default=0.2,
                      help="Hybrid polish pool fraction [0..1]")
    rail.add_argument("--rr_trials", type=int, default=1,
                      help="Randomized rounding trial count")
    # MIP-level controls
    rail.add_argument("--threads", type=int, default=1,
                      help="Solver threads (1 for single-core, 0 or omit = solver default)")
    rail.add_argument("--mip_gap", type=float, default=0.01,
                      help="Relative MIP gap target, e.g. 0.01 for 1%")
    # Gurobi comparison controls
    rail.add_argument("--gurobi_time_limit", type=float, default=30.0, 
                      help="Time limit for Gurobi comparison (seconds)")
    rail.add_argument("--gurobi_gap_limit", type=float, default=None,
                      help="Gap limit for Gurobi comparison (e.g., 0.01 for 1%)")
    rail.add_argument("--track_gurobi_anytime", action="store_true", default=True,
                      help="Track Gurobi's incumbent progression over time")

    rail.set_defaults(func=run_rail_cmd)

    # --- all subcommand (synthetic then RAIL) ---
    allp = sub.add_parser("all", help="Run synthetic (HYB+MIP) and then RAIL582")
    allp.add_argument("--rail_dir", type=str, default=None,
                      help="Path to OR-Library rail582 (or set env RAIL582_DIR)")
    allp.add_argument("--trials", type=int, default=20,
                      help="Number of independent trials to run")
    allp.add_argument("--seed", type=int, default=1221,
                      help="Base RNG seed (apps may offset per trial)")
    allp.add_argument("--solver", choices=["cbc", "gurobi"], default="cbc",
                      help="MIP solver backend for the combined runs (default: cbc)")
    # Hybrid knobs
    allp.add_argument("--polish_time", type=float, default=1)
    allp.add_argument("--polish_pool", type=float, default=0.2)
    allp.add_argument("--rr_trials", type=int, default=1)
    allp.add_argument("--improve_passes", type=int, default=1,
                      help="Local improvement passes (default: 1)")
    allp.add_argument("--threads", type=int, default=1,
                      help="Solver threads (default: 1)")
    allp.add_argument("--mip_gap", type=float, default=0.01,
                      help="Relative MIP gap target (default: 0.01)")
    # Gurobi comparison controls
    allp.add_argument("--gurobi_time_limit", type=float, default=30.0, 
                      help="Time limit for Gurobi comparison (seconds)")
    allp.add_argument("--gurobi_gap_limit", type=float, default=None,
                      help="Gap limit for Gurobi comparison (e.g., 0.01 for 1%)")
    allp.add_argument("--track_gurobi_anytime", action="store_true", default=True,
                      help="Track Gurobi's incumbent progression over time")
    allp.set_defaults(func=run_all_cmd)

    # --- suite subcommand (population + synthetic + rail) ---
    suite = sub.add_parser("suite", help="Run population_transport, synthetic set-cover, and RAIL582")
    suite.add_argument("--rail_dir", type=str, default=None,
                       help="Path to OR-Library rail582 (or set env RAIL582_DIR)")
    suite.add_argument("--out_dir", type=str, default="experiments/rail")
    suite.add_argument("--trials", type=int, default=20,
                       help="Number of independent trials to run (default: 10)")
    suite.add_argument("--seed", type=int, default=31,
                       help="Base RNG seed (apps may offset per trial)")
    suite.add_argument("--solver", choices=["cbc", "gurobi"], default="gurobi",
                       help="MIP solver backend for the full suite (default: cbc)")
    # Hybrid knobs
    suite.add_argument("--polish_time", type=float, default=0.5)
    suite.add_argument("--polish_pool", type=float, default=0.2)
    suite.add_argument("--rr_trials", type=int, default=1)
    suite.add_argument("--improve_passes", type=int, default=1,
                       help="Local improvement passes (default: 1)")
    suite.add_argument("--threads", type=int, default=1,
                       help="Solver threads (default: 1)")
    suite.add_argument("--mip_gap", type=float, default=0.01,
                       help="Relative MIP gap target (default: 0.01)")
    # Gurobi comparison controls
    suite.add_argument("--gurobi_time_limit", type=float, default=30.0, 
                       help="Time limit for Gurobi comparison (seconds)")
    suite.add_argument("--gurobi_gap_limit", type=float, default=None,
                       help="Gap limit for Gurobi comparison (e.g., 0.01 for 1%)")
    suite.add_argument("--track_gurobi_anytime", action="store_true", default=True,
                       help="Track Gurobi's incumbent progression over time")

    suite.set_defaults(func=run_suite_cmd)

# Add after the suite parser
    pop = sub.add_parser("pop", help="Run population and transport only")
    pop.add_argument("--out_dir", type=str, default="experiments/rail")
    pop.add_argument("--trials", type=int, default=10,
                     help="Number of independent trials to run (default: 3)")
    pop.add_argument("--seed", type=int, default=123,
                     help="Base RNG seed")
    pop.set_defaults(func=run_pop_cmd)
    
    # --- warmstart subcommand (compare Hybrid-only vs Gurobi-cold vs Gurobi-warm) ---
    warmstart = sub.add_parser("warmstart", 
                               help="Compare Hybrid-only vs Gurobi-cold vs Gurobi-warm")
    warmstart.add_argument("--out_dir", type=str, default="experiments/warmstart")
    warmstart.add_argument("--scales", type=str, default="",
                          help="Comma-separated n:m pairs, e.g. 200:4000,400:8000")
    warmstart.add_argument("--trials", type=int, default=10,
                          help="Number of trials per instance")
    warmstart.add_argument("--seed", type=int, default=42,
                          help="Random seed")
    warmstart.add_argument("--mip_timelimit", type=int, default=30,
                          help="Time limit for Gurobi (seconds)")
    warmstart.add_argument("--tau", type=float, default=0.1,
                          help="Entropy parameter tau")
    warmstart.add_argument("--tau_schedule", type=str, default=None,
                          help="Tau annealing schedule")
    warmstart.add_argument("--iters", type=int, default=50,
                          help="IPF iterations")
    warmstart.add_argument("--tol", type=float, default=1e-3,
                          help="IPF convergence tolerance")
    
    # In build_parser(), add these lines to the warmstart parser:
    warmstart.add_argument("--polish_time", type=float, default=1.0,
                          help="Hybrid polish time in seconds (default: 1.0)")
    warmstart.add_argument("--polish_pool", type=float, default=0.3,
                          help="Hybrid polish pool fraction (default: 0.3)")
    warmstart.add_argument("--rr_trials", type=int, default=5,
                          help="Randomized rounding trials (default: 5)")        
    warmstart.set_defaults(func=run_warmstart_cmd)



    # --- legacy apps (optional) ---
    apps = sub.add_parser("apps", help="Run population/transport experiments (legacy runner)")
    apps.add_argument("--config", type=str, default="")
    apps.set_defaults(func=run_apps_cmd)

    return p
# ------------------------------
# Main
# ------------------------------

# In your cli.py main function, fix the order:

def main():
    parser = build_parser()
    
    # Spyder F5: no args -> run everything
    #args = parser.parse_args(["all"]) if len(sys.argv) == 1 else parser.parse_args()
    #args = parser.parse_args(["rail"]) if len(sys.argv) == 1 else parser.parse_args()    
    args = parser.parse_args(["pop"]) if len(sys.argv) == 1 else parser.parse_args()
    args = parser.parse_args(["warmstart"]) if len(sys.argv) == 1 else parser.parse_args()
    
    # NOW override for warmstart experiments
# In main(), change the warmstart overrides to:
# In main(), change to:
# In main():
    if args.app == "warmstart":
        args.tau_schedule = "0.5,0.2,0.1"  # ← STOP HERE, don't go lower!
        args.iters = 60
        args.polish_time = 1.0    
#    if args.app == "warmstart":
#        args.tau_schedule = None  # ← Go back to single tau
#        args.iters = 50          # ← Back to 50
#        args.polish_time = 0.0   # ← Disable polish
#        print(f"[DEBUG] Using basic config: tau=0.1, iters=50, no polish")
        
#    if args.app == "warmstart":  # Only override when running warmstart
#        args.tau_schedule = "0.5,0.2,0.1,0.05,0.02,0.01"
#        args.iters = 60
#        print(f"[DEBUG] Overriding: tau_schedule={args.tau_schedule}, iters={args.iters}")
    
    # Now we can safely access args
    print(f"[DEBUG] track_gurobi_anytime from CLI: {getattr(args, 'track_gurobi_anytime', False)}")
    
    # Make the CLI choice visible to the app code
    import os
    os.environ["MIP_SOLVER"] = "gurobi"  # Force Gurobi for anytime tracking
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()