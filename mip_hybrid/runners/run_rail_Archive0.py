# mip_hybrid/runners/run_rail.py

import os
import argparse
import sys
import subprocess
import re
from datetime import datetime
import pandas as pd
import numpy as np

from ..io.writers import write_csv, write_latex, latex_summary_table

# ------------------------------------------------------------
# Aliases for reading app CSVs (varied column names)
# ------------------------------------------------------------
_ALIAS = {
    "hyb_int":      ["hyb_int", "hybrid_int", "hyb_obj", "hybrid_obj", "hyb_best", "int_best"],
    "mip_obj":      ["mip_obj", "mip_int", "mip_best", "mip_value"],
    "gap_pct":      ["gap_pct", "gap%", "gap_percent", "gap"],

    # Times
    "hyb_total":    ["hyb_total", "hyb_time", "hybrid_time", "T", "hyb_T", "hyb_total_time"],
    "polish_time":  ["polish_time", "polish", "polish_s"],
    "mip_time":     ["mip_time", "mip_t", "mip_total_time"],

    # Optional extras
    "relax_obj":    ["relax_obj", "lp_relax_obj", "relax_value", "relax"],
    "round_obj":    ["round_obj", "round_value", "round"],
    "polish_obj":   ["polish_obj", "polished_obj"],
    "relax_time":   ["relax_time"],
    "round_main":   ["round_main", "round_time"],
    "round_rr":     ["round_rr", "rr_time"],
    "improve":      ["improve", "improve_time"],
}

NEAR_OPT_PCT = 1.5  # threshold for labeling "near-optimum"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _first_present(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

def _extract_from_app_csv(app_csv_path: str) -> dict:
    """Read the app CSV and extract a single summary row (last row)."""
    if not os.path.exists(app_csv_path):
        return {}
    try:
        df = pd.read_csv(app_csv_path)
        if df.empty:
            return {}
        last = df.iloc[-1]
        out = {}
        for std, aliases in _ALIAS.items():
            col = _first_present(df.columns, aliases)
            if col is not None and pd.notna(last.get(col, np.nan)):
                out[std] = float(last[col])
        return out
    except Exception:
        return {}

def _run_module_cli(module: str, argv: list, cwd: str, log_path: str) -> int:
    """
    Run 'python -m {module} {argv...}' in a separate process, capture stdout/stderr to log_path.
    Return process return code.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Force UTF-8 for Windows consoles
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [sys.executable, "-X", "utf8", "-m", module] + argv

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("[cmd] " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        for line in proc.stdout:
            logf.write(line)
        ret = proc.wait()
    return ret

# ------------------------------------------------------------
# Log parser (rail_setcover output)
# ------------------------------------------------------------
def _parse_log_to_row(log_path: str) -> dict:
    """
    Parse key metrics from the rail_setcover console log.
    Robust to minor format changes; scans line by line.
    """
    row = {}
    if not os.path.exists(log_path):
        return row

    num = r"(\d+(?:\.\d+)?)"

    # MIP line
    mip_re       = re.compile(rf"MIP\s+obj\s*=\s*(?P<mip_obj>{num})\s*,\s*gap\s*=\s*(?P<mip_gap>{num})\s*,\s*t\s*=\s*(?P<mip_time>{num})s", re.I)

    # HYB block
    hyb_int_re   = re.compile(rf"HYB\s+int\s*=\s*(?P<hyb_int>{num})", re.I)
    hyb_T_re     = re.compile(rf"\bT\s*=\s*(?P<hyb_total>{num})s", re.I)  # total hybrid wall time
    gap_re       = re.compile(rf"gap%?\s*=\s*(?P<gap_pct>{num})", re.I)

    # objs: relax/round/polish
    relax_obj_re  = re.compile(rf"objs:.*?relax\s*=\s*(?P<relax_obj>{num})", re.I)
    round_obj_re  = re.compile(rf"objs:.*?round\s*=\s*(?P<round_obj>{num})", re.I)
    polish_obj_re = re.compile(rf"objs:.*?polish\s*=\s*(?P<polish_obj>{num})", re.I)

    # times(s): relax/round_main/round_rr/improve/polish/mip
    t_relax_re   = re.compile(rf"times\(s\).*?relax\s*=\s*(?P<relax_time>{num})", re.I)
    t_rmain_re   = re.compile(rf"times\(s\).*?round_main\s*=\s*(?P<round_main>{num})", re.I)
    t_rr_re      = re.compile(rf"times\(s\).*?round_rr\s*=\s*(?P<round_rr>{num})", re.I)
    t_impr_re    = re.compile(rf"times\(s\).*?improve\s*=\s*(?P<improve>{num})", re.I)
    t_polish_re  = re.compile(rf"times\(s\).*?polish\s*=\s*(?P<polish_time>{num})", re.I)
    t_mip_re     = re.compile(rf"times\(s\).*?mip\s*=\s*(?P<mip_time>{num})", re.I)

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = mip_re.search(line)
            if m:
                row["mip_obj"]  = float(m.group("mip_obj"))
                try:
                    row["mip_gap"]  = float(m.group("mip_gap"))
                except Exception:
                    pass
                row["mip_time"] = float(m.group("mip_time"))

            if "HYB" in line:
                m1 = hyb_int_re.search(line)
                if m1: row["hyb_int"] = float(m1.group("hyb_int"))

                m2 = hyb_T_re.search(line)
                if m2: row["hyb_total"] = float(m2.group("hyb_total"))

                m3 = gap_re.search(line)
                if m3: row["gap_pct"] = float(m3.group("gap_pct"))

                m4 = relax_obj_re.search(line)
                if m4: row["relax_obj"] = float(m4.group("relax_obj"))
                m5 = round_obj_re.search(line)
                if m5: row["round_obj"] = float(m5.group("round_obj"))
                m6 = polish_obj_re.search(line)
                if m6: row["polish_obj"] = float(m6.group("polish_obj"))

                t = t_relax_re.search(line)
                if t: row["relax_time"] = float(t.group("relax_time"))
                t = t_rmain_re.search(line)
                if t: row["round_main"] = float(t.group("round_main"))
                t = t_rr_re.search(line)
                if t: row["round_rr"] = float(t.group("round_rr"))
                t = t_impr_re.search(line)
                if t: row["improve"] = float(t.group("improve"))
                t = t_polish_re.search(line)
                if t: row["polish_time"] = float(t.group("polish_time"))
                t = t_mip_re.search(line)
                if t: row["mip_time"] = float(t.group("mip_time"))

    return row

# ------------------------------------------------------------
# Main runner API
# ------------------------------------------------------------
def run_rail(rail_dir: str = None, synthetic_sizes=None, out_dir: str = "experiments/rail"):
    os.makedirs(out_dir, exist_ok=True)

    stamp    = _ts()
    log_path = os.path.join(out_dir, f"rail_run_{stamp}.log")
    app_csv  = os.path.join(out_dir, f"rail_results_{stamp}.csv")

    # Decide which app to run
    if synthetic_sizes:
        module = "mip_hybrid.apps.synth_setcover"
        scales = ",".join(f"{n}x{m}" for (n, m, *_rest) in synthetic_sizes)
        argv   = ["--out", app_csv, "--scales", scales, "--trials", "1",
                  "--tau", "0.1", "--with_mip", "--mip_timelimit", "30"]
    elif rail_dir:
        module = "mip_hybrid.apps.rail_setcover"
        argv   = ["--out", app_csv, "--rail_path", rail_dir,
                  "--tau", "0.1", "--rounding", "dual", "--timelimit", "60"]
    else:
        module = "mip_hybrid.apps.synth_setcover"
        argv   = ["--out", app_csv, "--scales", "200x4000", "--trials", "1",
                  "--tau", "0.1", "--with_mip", "--mip_timelimit", "30"]

    ret = _run_module_cli(module, argv, cwd=os.getcwd(), log_path=log_path)
    if ret != 0:
        print(f"[warn] {module} exited with code {ret}. See log: {log_path}")

    # Prefer app CSV; otherwise parse the log
    parsed = _extract_from_app_csv(app_csv) or _parse_log_to_row(log_path)

    # Augment with metadata
    parsed.update({
        "family": "RAIL582" if rail_dir else "SYN",
        "scales": ",".join(f"{n}x{m}" for (n, m, *_r) in (synthetic_sizes or [])),
        "log_path": log_path,
        "app_csv": app_csv if os.path.exists(app_csv) else "",
    })
    df = pd.DataFrame([parsed])

    # Compute HYB→MIP gap and “near-opt time”
    def _safe_gap(hyb, mip):
        if mip is None or pd.isna(mip) or abs(mip) < 1e-9 or hyb is None or pd.isna(hyb):
            return np.nan
        return 100.0 * max(0.0, (hyb - mip) / abs(mip))

    df["hyb_gap_pct"] = df.apply(lambda r: _safe_gap(r.get("hyb_int"), r.get("mip_obj")), axis=1)

    # Elapsed time to first near-optimal incumbent: sum of controller stages up to polish
    rmain   = float(df.get("round_main", np.nan))
    rrr     = float(df.get("round_rr",   np.nan))
    impr    = float(df.get("improve",    np.nan))
    polish  = float(df.get("polish_time",np.nan))
    comp_sum = sum(v for v in (rmain, rrr, impr, polish) if not np.isnan(v))

    df["hyb_nearopt_time"] = np.where(
        (df["hyb_gap_pct"].notna()) & (df["hyb_gap_pct"] <= NEAR_OPT_PCT) & (comp_sum > 0),
        comp_sum,
        np.nan,
    )

    # Write one-row summary CSV
    csv_path = write_csv(df, out_dir, stem="rail_results_summary")

    # -------------------- LaTeX: compact "story" table --------------------
    story_metrics = {
        "hyb_gap_pct":      ["median"],
        "hyb_nearopt_time": ["median"],
    }
    story_rename = {
        "hyb_gap_pct_median":      "HYB\\,$\\to$\\,MIP gap (\\%)",
        "hyb_nearopt_time_median": "HYB near-opt (s)",
    }
    story_order = ["family", "HYB\\,$\\to$\\,MIP gap (\\%)", "HYB near-opt (s)"]

    tex_story = latex_summary_table(
        df,
        group_cols=["family"],
        metrics=story_metrics,
        caption="RAIL / Hybrid and MIP summary with time to near-optimum (CSV preferred, log fallback).",
        label="tab:rail_log_summary",
        col_order=story_order,
        col_rename=story_rename,
        escape_text=True,
    )
    tex_story_path = write_latex(tex_story, out_dir, stem="rail_summary_table_story")

    # -------------------- LaTeX: full breakdown table --------------------
    base_labels = {
        "hyb_int":          "HYB int",
        "mip_obj":          "MIP obj",
        "gap_pct":          "Gap (\\%)",
        "hyb_gap_pct":      "HYB\\,$\\to$\\,MIP gap (\\%)",
        "hyb_nearopt_time": "HYB near-opt (s)",
        "hyb_total":        "HYB total (s)",
        "polish_time":      "Polish (s)",
        "mip_time":         "MIP time (s)",
        # optional extras:
        "relax_obj":        "Relax obj",
        "round_obj":        "Round obj",
        "polish_obj":       "Polish obj",
        "relax_time":       "Relax (s)",
        "round_main":       "Round main (s)",
        "round_rr":         "Round RR (s)",
        "improve":          "Improve (s)",
    }
    present     = [c for c in base_labels if c in df.columns]
    if present:
        metrics_full    = {c: ["median"] for c in present}
        col_rename_full = {f"{c}_median": base_labels[c] for c in present}
        col_order_full  = ["family"] + [base_labels[c] for c in present]
    else:
        metrics_full    = {"log_path": ["count"]}
        col_rename_full = {"log_path_count": "runs"}
        col_order_full  = ["family", "runs"]

    tex_full = latex_summary_table(
        df,
        group_cols=["family"],
        metrics=metrics_full,
        caption="RAIL / Hybrid and MIP detailed summary (CSV preferred, log fallback).",
        label="tab:rail_log_summary_full",
        col_order=col_order_full,
        col_rename=col_rename_full,
        escape_text=True,
    )
    tex_full_path = write_latex(tex_full, out_dir, stem="rail_summary_table_full")

    # Console banner (Spyder-friendly)
    hyb     = float(df.get("hyb_int", np.nan))
    mip     = float(df.get("mip_obj", np.nan))
    hyb_tot = float(df.get("hyb_total", np.nan))
    pol     = float(df.get("polish_time", np.nan))
    gap     = float(df.get("hyb_gap_pct", np.nan))
    near    = float(df.get("hyb_nearopt_time", np.nan))
    if not np.isnan(hyb):
        parts = [f"[summary] HYB {hyb:.3f}"]
        if not np.isnan(mip):
            parts.append(f"vs MIP {mip:.3f} (gap {gap:.2f}%)")
        if not np.isnan(near):
            parts.append(f"near-opt in {near:.2f}s")
        if not np.isnan(hyb_tot):
            parts.append(f"total {hyb_tot:.2f}s")
        if not np.isnan(pol):
            parts.append(f"polish {pol:.2f}s")
        if not np.isnan(gap) and gap <= NEAR_OPT_PCT and not np.isnan(near):
            parts.append("🔥")
        print(" | ".join(parts))

    return {
        "csv": csv_path,
        "tex_story": tex_story_path,
        "tex_full": tex_full_path,
        "log": log_path,
        "app_csv": parsed.get("app_csv", ""),
    }

# ------------------------------------------------------------
# CLI entry (optional)
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rail_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="experiments/rail")
    p.add_argument("--syn", type=str, default="", help="n:m pairs, e.g. 400:200,800:400")
    args = p.parse_args()

    synthetic = []
    if args.syn:
        for token in args.syn.split(","):
            n, m = token.split(":")
            synthetic.append((int(n), int(m), 1, 0.0, 42))

    run_rail(args.rail_dir, synthetic, args.out_dir)

if __name__ == "__main__":
    main()
