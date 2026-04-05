
"""
write_trace_from_log.py

Usage (now easier):
    # Just run it — it will pick the latest *_run_*.log in experiments/rail
    python write_trace_from_log.py

    # Or specify a directory to search
    python write_trace_from_log.py --dir experiments/rail

    # Or specify an exact log path and optional output path
    python write_trace_from_log.py --log experiments/rail/rail_run_20250924_093629.log --out experiments/rail/trace_custom.csv

Output CSV columns:
    t_sec, phase, incumbent, bound, rel_gap, note
"""
import argparse, csv, math, re, sys
from pathlib import Path

def parse_trace(log_path: Path):
    txt = log_path.read_text(encoding="utf-8", errors="ignore")

    mip_line = re.search(r"\[rail\]\s*MIP obj\s*=\s*([0-9.]+),\s*gap\s*=\s*([0-9.]+),\s*t\s*=\s*([0-9.]+)s", txt)
    hyb_line = re.search(
        r"\[rail\]\s*HYB.*?objs:\s*relax=([0-9.]+),\s*round=([0-9.]+),\s*polish=([0-9.]+),\s*mip=([0-9.]+),\s*gap%=\s*([0-9.]+)\s*\|\s*times\(s\):\s*relax=([0-9.]+),\s*round_main=([0-9.]+),\s*round_rr=([0-9.]+),\s*improve=([0-9.]+),\s*polish=([0-9.]+),\s*mip=([0-9.]+)",
        txt
    )

    rows = []
    if hyb_line:
        relax_obj, round_obj, polish_obj, mip_obj_for_ref, hyb_gap_pct, t_relax, t_round_main, t_round_rr, t_improve, t_polish, t_mip = hyb_line.groups()
        relax_obj = float(relax_obj); round_obj = float(round_obj); polish_obj = float(polish_obj)
        t_relax = float(t_relax); t_round_main = float(t_round_main); t_round_rr = float(t_round_rr)
        t_improve = float(t_improve); t_polish = float(t_polish)

        t_round = t_relax + t_round_main + t_round_rr
        t_after_improve = t_round + t_improve
        t_after_polish = t_after_improve + t_polish

        rows.append((t_relax, "relax", relax_obj, math.nan, math.nan, "HYB relax"))
        rows.append((t_round, "round", round_obj, math.nan, math.nan, "HYB round (main+rr)"))
        if t_improve > 0:
            rows.append((t_after_improve, "improve", round_obj, math.nan, math.nan, "HYB improve"))
        rows.append((t_after_polish, "polish", polish_obj, math.nan, math.nan, "HYB polish"))

    if mip_line:
        mip_obj = float(mip_line.group(1))
        mip_gap = float(mip_line.group(2))
        mip_t = float(mip_line.group(3))
        if mip_gap <= 0.0:
            bound = mip_obj
        else:
            bound = mip_obj * (1.0 - mip_gap)
        rel_gap = abs(mip_obj - bound) / max(1e-12, abs(mip_obj))
        rows.append((mip_t, "mip", mip_obj, bound, rel_gap, "MIP final"))

    rows.sort(key=lambda r: r[0])
    return rows

def find_latest_log(search_dir: Path) -> Path | None:
    candidates = list(search_dir.glob("*_run_*.log"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="", help="Path to a specific *_run_*.log (optional)")
    ap.add_argument("--out", default="", help="Output CSV path (optional)")
    ap.add_argument("--dir", default="experiments/rail", help="Directory to search for latest *_run_*.log (default: experiments/rail)")
    args = ap.parse_args()

    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            print(f"[trace] Log not found: {log_path}", file=sys.stderr)
            sys.exit(2)
    else:
        search_dir = Path(args.dir)
        if not search_dir.exists():
            print(f"[trace] Search dir not found: {search_dir}", file=sys.stderr)
            sys.exit(2)
        latest = find_latest_log(search_dir)
        if latest is None:
            print(f"[trace] No *_run_*.log files found in: {search_dir}", file=sys.stderr)
            sys.exit(2)
        log_path = latest
        print(f"[trace] Using latest log: {log_path}")

    rows = parse_trace(log_path)
    if not rows:
        print("[trace] No traceable lines found in the log. Is it a HYB/MIP run?", file=sys.stderr)
        sys.exit(2)

    if args.out:
        out_csv = Path(args.out)
    else:
        m = re.search(r"(\d{8}_\d{6})", log_path.name)
        stamp = m.group(1) if m else "trace"
        out_csv = log_path.parent / f"trace_{stamp}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "phase", "incumbent", "bound", "rel_gap", "note"])
        for r in rows:
            w.writerow(r)

    print(f"[trace] wrote {out_csv} rows={len(rows)}")

if __name__ == "__main__":
    main()
