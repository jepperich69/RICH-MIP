# -*- coding: utf-8 -*-
"""
Final RAIL runner (drop-in).
- Calls mip_hybrid.apps.rail_setcover in-process (uses main_from_cli_namespace).
- ALWAYS writes: main CSV, .log, stats CSV, and three TeX snippets (story/full/detailed).
- Robust: creates placeholders if post-processing fails, so CLI prints never show blanks.
- Fast defaults baked in: rr_trials=1, improve_passes=0, polish_time=3.0, polish_pool=0.3 (unless overridden).
- No pandas dependency.
"""
import os, sys, csv, math, datetime as dt
from contextlib import contextmanager

# ------------------------------
# Utilities
# ------------------------------
def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

@contextmanager
def tee_stdout(log_path: str):
    """Tee stdout to both console and a log file."""
    _ensure_dir(os.path.dirname(log_path))
    old = sys.stdout
    class Tee:
        def __init__(self, path):
            self.f = open(path, "w", encoding="utf-8")
            self.old = old
        def write(self, s):
            try: self.old.write(s)
            except Exception: pass
            try: self.f.write(s)
            except Exception: pass
        def flush(self):
            try: self.old.flush()
            except Exception: pass
            try: self.f.flush()
            except Exception: pass
        def close(self):
            try: self.f.close()
            except Exception: pass
    t = Tee(log_path)
    sys.stdout = t
    try:
        yield
    finally:
        sys.stdout = old
        t.close()

def _safe_float(v, default=math.nan):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default

def _read_rows(csv_path: str):
    if not csv_path or (not os.path.isfile(csv_path)):
        return []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            return list(rdr)
    except Exception:
        return []

def write_stats_csv(csv_in: str, stats_out: str):
    """Create a simple stats summary CSV from the main results CSV"""
    try:
        rows = read_rows(csv_in)
        if not rows:
            return
        
        # Create a simple placeholder stats file
        with open(stats_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['total_experiments', str(len(rows))])
            
            # Calculate some basic stats if data exists
            hyb_times = [_safe_float(r.get('hyb_time', r.get('hyb_total', 0))) for r in rows]
            mip_times = [_safe_float(r.get('mip_time', 0)) for r in rows]
            
            if hyb_times:
                writer.writerow(['avg_hyb_time', f"{sum(hyb_times)/len(hyb_times):.3f}"])
            if mip_times:
                writer.writerow(['avg_mip_time', f"{sum(mip_times)/len(mip_times):.3f}"])
                
    except Exception as e:
        # Create minimal placeholder if anything fails
        with open(stats_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['status', 'processing_failed'])

def ensure_file(file_path: str, text: str = ""):
    """Ensure a file exists with given content"""
    try:
        ensure_dir(os.path.dirname(file_path))
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
    except Exception:
        pass

def _write_stats_csv(csv_in: str, csv_out: str) -> None:
    rows = _read_rows(csv_in)
    _ensure_dir(os.path.dirname(csv_out))
    if not rows:
        with open(csv_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["metric", "value"]); w.writerow(["note", "no rows"])
        return

    hyb_ints   = [_safe_float(r.get("hyb_int"))    for r in rows]
    mip_objs   = [_safe_float(r.get("mip_obj"))    for r in rows]
    gaps       = [_safe_float(r.get("gap_pct"))    for r in rows]
    hyb_total  = [_safe_float(r.get("hyb_total"))  for r in rows]
    polish_t   = [_safe_float(r.get("polish_time")) for r in rows]
    mip_time   = [_safe_float(r.get("mip_time"))   for r in rows]

    def _min(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return min(xs) if xs else math.nan
    def _avg(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return sum(xs) / len(xs) if xs else math.nan

    stats = [
        ("n_rows", len(rows)),
        ("hyb_int_min", _min(hyb_ints)),
        ("hyb_int_avg", _avg(hyb_ints)),
        ("mip_obj_min", _min(mip_objs)),
        ("mip_obj_avg", _avg(mip_objs)),
        ("gap_pct_avg", _avg(gaps)),
        ("hyb_total_avg_s", _avg(hyb_total)),
        ("polish_time_avg_s", _avg(polish_t)),
        ("mip_time_avg_s", _avg(mip_time)),
    ]
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats:
            w.writerow([k, v])

def _tex_escape(s: str) -> str:
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))

def read_rows(csv_path: str):
    """Read CSV file and return list of dictionaries"""
    rows = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            pass
    return rows

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def write_tex_files(csv_in: str, story_path: str, full_path: str, detailed_path: str) -> None:
    rows = read_rows(csv_in)
    ensure_dir(os.path.dirname(story_path))
    
    # Detect format by checking if synthetic or RAIL data
    is_synthetic = False
    if rows and 'family' in rows[0] and rows[0].get('family') == 'SYN':
        is_synthetic = True
    
    # STORY
    try:
        best_hyb = min(_safe_float(r.get("hyb_int")) for r in rows) if rows else math.nan
    except Exception:
        best_hyb = math.nan
    try:
        best_mip = min(_safe_float(r.get("mip_obj")) for r in rows) if rows else math.nan
    except Exception:
        best_mip = math.nan
    try:
        # Use appropriate time column based on format
        time_col = "hyb_time" if is_synthetic else "hyb_total"
        avg_time = sum(_safe_float(r.get(time_col)) for r in rows) / max(1, len(rows)) if rows else math.nan
    except Exception:
        avg_time = math.nan
    
    story = "\\begin{quote}\n" \
            f"Best HYB={best_hyb:.0f}, best MIP={best_mip:.0f}, avg HYB time={avg_time:.2f}s.\n" \
            "\\end{quote}\n"
    with open(story_path, "w", encoding="utf-8") as f:
        f.write(story)
    
    # FULL
    if is_synthetic:
        cols = ["trial", "hyb_int", "mip_obj", "gap_pct", "hyb_time", "mip_time"]
    else:
        cols = ["trial", "hyb_int", "mip_obj", "gap_pct", "hyb_total", "polish_time"]
    
    header = " & ".join(cols) + r" \\ \hline" + "\n"
    lines = []
    for r in rows:
        line = " & ".join(_tex_escape(str(r.get(c, ""))) for c in cols) + r" \\"
        lines.append(line)
    content = "\\begin{tabular}{lrrrrr}\n\\hline\n" + header + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n"
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # DETAILED
    if is_synthetic:
        dcols = ["n", "m", "trial", "density", "hyb_int", "mip_obj", "gap_pct", "hyb_time", "relax_time", "mip_time"]
        colspec = "lrrrrrrrrr"
    else:
        dcols = ["n","m","k","seed","controller","relax_obj","round_obj","polish_obj","mip_obj",
                 "gap_pct","relax_time","round_main","round_rr","improve","polish_time","mip_time"]
        colspec = "lrrrrlrrrrrrrrrr"
    
    dheader = " & ".join(dcols) + r" \\ \hline" + "\n"
    dlines = []
    for r in rows:
        line = " & ".join(_tex_escape(str(r.get(c, ""))) for c in dcols) + r" \\"
        dlines.append(line)
    dcontent = f"\\begin{{tabular}}{{{colspec}}}\\n\\hline\\n" + dheader + "\n".join(dlines) + "\n\\hline\n\\end{tabular}\n"
    with open(detailed_path, "w", encoding="utf-8") as f:
        f.write(dcontent)

def _ensure_file(path: str, text: str = "") -> None:
    try:
        _ensure_dir(os.path.dirname(path))
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
    except Exception:
        pass

# ------------------------------
# Call app in-process
# ------------------------------
def _call_app_inproc(rail_path, out_csv, out_dir, **kwargs) -> bool:
    """
    Try to call the appropriate app in-process using main_from_cli_namespace.
    Returns True on success, False if the entrypoint is not found.
    
    Logic:
    - If rail_path is None: call synthetic set cover generator
    - If rail_path points to RAIL582 data: call rail_setcover app
    - Otherwise: call rail_setcover app (default)
    """
    try:
        if rail_path is None:
            # Synthetic case - call synthetic set cover
            print("[runner] Using synthetic set cover generator")
            from mip_hybrid.apps import synth_setcover as app
            
            # Convert kwargs to synth_setcover format
            scales = kwargs.get('scales', '200x4000')  # default if not provided
            trials = kwargs.get('trials', 1)

            print(f"[DEBUG] scales parameter: {scales}")
            print(f"[DEBUG] trials parameter: {kwargs.get('trials', 1)}")
            
            # Call synth_setcover directly with appropriate args
            import argparse
            import sys
            
            # Build args for synth_setcover
            old_argv = sys.argv[:]
            try:
                sys.argv = [
                    'synth_setcover',
                    '--scales', str(scales),
                    '--trials', str(trials),
                    '--out', str(out_csv),
                    '--seed', str(kwargs.get('seed', 42)),
                    '--solver', 'gurobi',
                    '--tau', str(kwargs.get('tau', 0.1)),
                    '--iters', str(kwargs.get('iters', 500)),  # Add this line
                    '--with_mip'
                ]                
                app.main()
                return True
            finally:
                sys.argv = old_argv
                
        else:
            # RAIL582 case - call rail_setcover
            print(f"[runner] Using RAIL582 set cover app with path: {rail_path}")
            from mip_hybrid.apps import rail_setcover as app
            
            if not hasattr(app, "main_from_cli_namespace"):
                print("[runner] rail_setcover.main_from_cli_namespace not found")
                return False
                
            app.main_from_cli_namespace(
                rail_path=rail_path,
                out=out_csv,
                out_dir=out_dir,
                **kwargs
            )
            return True
            
    except Exception as e:
        print(f"[runner] app call failed: {e}")
        return False
# ------------------------------
# Core API
# ------------------------------
def run_rail(
    rail_dir,
    syn_list,
    out_dir,
    trials=1,
    seed=None,
    solver="gurobi",
    polish_time=3.0,
    polish_pool=0.3,
    rr_trials=None,
    # Optional knobs (pass-throughs to app)
    tau=0.1,
    tau_schedule=None,
    timelimit=60.0,
    kcover=1,
    controller="both",
    iters=200,
    tol=1e-3,
    rounding="dual",
    lp_lower_bound=False,
    profile=False,
    profile_out=None,
    rr_thr_max=0.8,
    improve_passes=1,
    threads=1,
    mip_gap=0.01,
    gurobi_time_limit=None,
    gurobi_gap_limit=None,
    track_gurobi_anytime=False,
):
    """
    Always produces:
      - main CSV
      - .log
      - stats CSV
      - story/full/detailed TeX
    Paths are returned in a dict. Files exist even if placeholders are needed.
    """
    print(f"[DEBUG] run_rail called with: gurobi_time_limit={gurobi_time_limit}, gurobi_gap_limit={gurobi_gap_limit}, track_gurobi_anytime={track_gurobi_anytime}")
    ensure_dir(out_dir)

    # ---- Fast defaults (runner-level) ----
    rr_trials = 1 if rr_trials is None else int(rr_trials)
    improve_passes = 0 if improve_passes is None else int(improve_passes)
    polish_time = 3.0 if polish_time is None else float(polish_time)
    polish_pool = 0.3 if polish_pool is None else float(polish_pool)

    # ---------- (A) Synthetic branch ----------
    if syn_list:
        ts = _ts()
        syn_csv  = os.path.join(out_dir, f"syn_results_{ts}.csv")
        syn_log  = os.path.join(out_dir, f"syn_run_{ts}.log")
        print("\n[all] Synthetic sweep (HYB + MIP)")

        # Build scales string like "200x4000,400x8000"
        scales_tokens = []
        for item in syn_list:
            try:
                n, m = int(item[0]), int(item[1])
                scales_tokens.append(f"{n}x{m}")
            except Exception:
                continue
        scales_str = ",".join(scales_tokens) if scales_tokens else "200x4000"

        with tee_stdout(syn_log):
            ok = _call_app_inproc(
                rail_path=None,
                out_csv=syn_csv,
                out_dir=out_dir,
                trials=trials,
                solver=solver,
                polish_time=polish_time,
                polish_pool=polish_pool,
                rr_trials=rr_trials,
                seed=seed,
                scales=scales_str,
                tau=tau,
                tau_schedule=tau_schedule,
                timelimit=timelimit,
                kcover=kcover,
                controller=controller,
                iters=iters,
                tol=tol,
                rounding=rounding,
                lp_lower_bound=lp_lower_bound,
                profile=profile,
                profile_out=profile_out,
                rr_thr_max=rr_thr_max,
                improve_passes=improve_passes,
                threads=threads,
                mip_gap=mip_gap,gurobi_time_limit=gurobi_time_limit,
                gurobi_gap_limit=gurobi_gap_limit,
                track_gurobi_anytime=track_gurobi_anytime              
            )
            if not ok:
                # write a tiny placeholder CSV so downstream always has something
                with open(syn_csv, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["family","note"])
                    w.writerow(["setcover","in-process entrypoint not found"])

        # Post-process (robust)
        syn_stats = os.path.join(out_dir, f"syn_results_stats_{ts}.csv")
        write_stats_csv(syn_csv, syn_stats)

        syn_story = os.path.join(out_dir, f"syn_summary_table_story_{ts}.tex")
        syn_full  = os.path.join(out_dir, f"syn_summary_table_full_{ts}.tex")
        syn_det   = os.path.join(out_dir, f"syn_summary_table_detailed_{ts}.tex")
        write_tex_files(syn_csv, syn_story, syn_full, syn_det)

        # Ensure log exists even if tee failed
        ensure_file(syn_log, text="% synthetic log placeholder\n")

        return {
            "csv": syn_csv,
            "csv_stats": syn_stats,
            "tex_story": syn_story,
            "tex_full": syn_full,
            "tex_detailed": syn_det,
            "log": syn_log,
        }

    # ---------- (B) RAIL582 branch ----------
    if rail_dir:
        ts = _ts()
        out_csv  = os.path.join(out_dir, f"rail582_results_{ts}.csv")
        log_path = os.path.join(out_dir, f"rail582_run_{ts}.log")
        print(f"[rail] RAIL582 path: {rail_dir}")
        print(f"[rail] RAIL 582 (HYB + MIP) → {out_csv}")

        with tee_stdout(log_path):
            ok = _call_app_inproc(
                rail_path=rail_dir,
                out_csv=out_csv,
                out_dir=out_dir,
                trials=trials,
                solver=solver,
                polish_time=polish_time,
                polish_pool=polish_pool,
                rr_trials=rr_trials,
                seed=seed,
                tau=tau,
                tau_schedule=tau_schedule,
                timelimit=timelimit,
                kcover=kcover,
                controller=controller,
                iters=iters,
                tol=tol,
                rounding=rounding,
                lp_lower_bound=lp_lower_bound,
                profile=profile,
                profile_out=profile_out,
                rr_thr_max=rr_thr_max,
                improve_passes=improve_passes,
                threads=threads,
                mip_gap=mip_gap,
                gurobi_time_limit=gurobi_time_limit,
                gurobi_gap_limit=gurobi_gap_limit,
                track_gurobi_anytime=track_gurobi_anytime
            )
            if not ok:
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["family","note"])
                    w.writerow(["RAIL582","in-process entrypoint not found"])

        # Post-process (robust)
        stats_csv_path = os.path.join(out_dir, f"rail_results_stats_{ts}.csv")
        _write_stats_csv(out_csv, stats_csv_path)

        tex_story_path = os.path.join(out_dir, f"rail_summary_table_story_{ts}.tex")
        tex_full_path  = os.path.join(out_dir, f"rail_summary_table_full_{ts}.tex")
        tex_detailed_path = os.path.join(out_dir, f"rail_summary_table_detailed_{ts}.tex")
        write_tex_files(out_csv, tex_story_path, tex_full_path, tex_detailed_path)

        ensure_file(log_path, text="% rail log placeholder\n")

        return {
            "csv": out_csv,
            "csv_stats": stats_csv_path,
            "tex_story": tex_story_path,
            "tex_full": tex_full_path,
            "tex_detailed": tex_detailed_path,
            "log": log_path,
        }

    # ---------- (C) nothing to do ----------
    return {"csv": "", "csv_stats": "", "tex_story": "", "tex_full": "", "tex_detailed": "", "log": ""}


def run_one_app(module, argv, family, out_dir):
    """Run a standalone app module with given argv"""
    try:
        import importlib
        import sys
        app_module = importlib.import_module(module)
        
        old_argv = sys.argv[:]
        try:
            sys.argv = [module] + argv
            app_module.main()
            return {"csv": argv[argv.index("--out") + 1] if "--out" in argv else ""}
        finally:
            sys.argv = old_argv

    except Exception as e:
        print(f"[runner] Failed to run {module}: {e}")
        return {"csv": ""}

# ------------------------------
# Minimal CLI for ad-hoc runs
# ------------------------------
def main():
    res = None
    import argparse
    p = argparse.ArgumentParser(description="Final RAIL runner")
    p.add_argument("--rail_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="experiments/rail")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--solver", type=str, default="gurobi", choices=["cbc","gurobi"])
    p.add_argument("--syn", type=str, default="", help="e.g. 200x4000,400x8000")
    p.add_argument("--polish_time", type=float, default=3.0)
    p.add_argument("--polish_pool", type=float, default=0.3)
    p.add_argument("--rr_trials", type=int, default=1)          # fast default
    p.add_argument("--improve_passes", type=int, default=0)     
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--mip_gap", type=float, default=0.01)
# fast default
    args = p.parse_args()
    
    print(f"[DEBUG] Received with_mip: {args.with_mip}")  # Add this line
    print(f"[DEBUG] Output path: {args.out}")  # Add this line

    syn_list = []
    if args.syn:
        for tok in args.syn.split(","):
            try:
                n, m = tok.split("x")
                syn_list.append((int(n), int(m), 1, 0.0, 42))
            except Exception:
                pass

    res = run_rail(
        rail_dir=args.rail_dir,
        syn_list=syn_list if syn_list else None,
        out_dir=args.out_dir,
        trials=args.trials,
        seed=args.seed,
        solver=args.solver,
        polish_time=args.polish_time,
        polish_pool=args.polish_pool,
        rr_trials=args.rr_trials,
        improve_passes=args.improve_passes,
        threads=args.threads,
        mip_gap=args.mip_gap,
    )

if __name__ == "__main__":
    main()
