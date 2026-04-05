# mip_hybrid/io/writers.py
import os
from datetime import datetime
from typing import Dict, List, Iterable, Optional, Union, Callable

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Filesystem / CSV helpers
# ----------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    """Return a filesystem-friendly timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def write_csv(df: pd.DataFrame, out_dir: str, stem: str) -> str:
    """Write DataFrame to CSV in out_dir with timestamp suffix."""
    ensure_dir(out_dir)
    fp = os.path.join(out_dir, f"{stem}_{timestamp()}.csv")
    df.to_csv(fp, index=False)
    return fp

def write_latex(tex: str, out_dir: str, stem: str) -> str:
    """Write LaTeX string to a .tex file in out_dir with timestamp."""
    ensure_dir(out_dir)
    fp = os.path.join(out_dir, f"{stem}_{timestamp()}.tex")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(tex)
    return fp

# ----------------------------------------------------------------------
# LaTeX utilities
# ----------------------------------------------------------------------

_LATEX_ESCAPE_MAP = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}

def escape_latex(text: Union[str, float, int]) -> str:
    """Lightweight LaTeX escaping for cell text."""
    s = "" if text is None else str(text)
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPE_MAP.get(ch, ch))
    return "".join(out)

def _format_float(x: float, fmt: str) -> str:
    """Format a float using either printf-style or Python format specifiers."""
    if np.isnan(x):
        return ""
    # Support ':.3g', '.2f', etc.
    if fmt.startswith(":"):
        return format(x, fmt[1:])
    if fmt.startswith("%"):  # e.g., "%.3f"
        return (fmt % x)
    return format(x, fmt)

def _flatten_agg_columns(cols: Iterable) -> List[str]:
    """Flatten MultiIndex columns produced by groupby.agg."""
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            base = "_".join([str(k) for k in c if k != ""])
            flat.append(base)
        else:
            flat.append(str(c))
    return flat

def _bold_best_per_group(df: pd.DataFrame,
                         group_cols: List[str],
                         metric_cols: List[str],
                         mode: str = "min") -> pd.DataFrame:
    """
    Return a copy of df where best values in metric_cols are wrapped in \textbf{...}
    within each group defined by group_cols.
    """
    assert mode in ("min", "max")
    out = df.copy()
    # Work on numeric comparable copies while keeping string rendering
    for g_vals, idx in out.groupby(group_cols, dropna=False).groups.items():
        block = out.loc[idx, metric_cols]
        # Convert to numeric where possible
        block_num = block.apply(pd.to_numeric, errors="coerce")
        if mode == "min":
            mask = (block_num == block_num.min(skipna=True))
        else:
            mask = (block_num == block_num.max(skipna=True))
        # Bold the corresponding string cells
        for r, c in zip(*np.where(mask.values)):
            rr = block.index[r]
            cc = metric_cols[c]
            val = out.at[rr, cc]
            if isinstance(val, str) and val.startswith(r"\textbf{"):
                continue
            out.at[rr, cc] = rf"\textbf{{{val}}}"
    return out

# ----------------------------------------------------------------------
# Main: build a compact Overleaf-ready table from a DataFrame
# ----------------------------------------------------------------------

def latex_summary_table(
    df: pd.DataFrame,
    group_cols: List[str],
    metrics: Dict[str, Union[str, List[Union[str, Callable]]]],
    caption: str,
    label: str,
    *,
    col_order: Optional[List[str]] = None,
    col_rename: Optional[Dict[str, str]] = None,
    float_format: str = ":.3g",
    bold_best: Optional[Dict[str, str]] = None,
    # e.g., {"obj_median": "min", "time_total_median": "min"}; per-column mode
    placement: str = "t",
    use_small: bool = True,
    resize_to_textwidth: bool = False,
    colspec: Optional[str] = None,
    escape_text: bool = True,
    note: Optional[str] = None,
) -> str:
    r"""
    Create a compact Overleaf-ready \small table summarizing df via groupby-agg.

    Parameters
    ----------
    df : DataFrame
    group_cols : list[str]
        Columns to group rows by (these appear as left-most columns).
    metrics : dict
        {column: agg or [aggs]} passed to groupby().agg(...).
        Example: {"obj": "median", "time_total": ["median", "max"]}
    caption, label : str
        LaTeX caption and \label.
    col_order : list[str], optional
        Exact column order for the final table (after flattening/renaming).
    col_rename : dict, optional
        Rename columns (after flattening) to display-friendly labels.
    float_format : str, default ':.3g'
        Format for float cells (printf or format-spec, e.g., '%.2f' or ':.3g').
    bold_best : dict[str, 'min'|'max'], optional
        Columns in which to bold the best values (per group). Keys must be final column names.
    placement : str, default 't'
        LaTeX table float placement (e.g., 't', 'h', '!ht').
    use_small : bool, default True
        Prefix table with \small.
    resize_to_textwidth : bool, default False
        Wrap tabular in \resizebox{\textwidth}{!}{...}.
    colspec : str, optional
        Override tabular column spec (e.g., 'lrrrr'). If None, auto 'l'*ncols.
    escape_text : bool, default True
        Apply LaTeX escaping to non-math text.
    note : str, optional
        Optional note under the table in \footnotesize.

    Returns
    -------
    str : LaTeX code.
    """
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dict")

    # 1) Aggregate
    agg = df.groupby(group_cols, dropna=False).agg(metrics)
    agg.columns = _flatten_agg_columns(agg.columns)
    agg = agg.reset_index()

    # 2) Formatting numerics
    formatted = agg.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: _format_float(x, float_format))

    # 3) Optional renames before ordering/bolding
    if col_rename:
        formatted = formatted.rename(columns=col_rename)

    # 4) Optional column order
    if col_order:
        missing = [c for c in col_order if c not in formatted.columns]
        if missing:
            raise KeyError(f"col_order contains columns not in table: {missing}")
        formatted = formatted[col_order]

    # 5) Optional bolding of best columns per group
    if bold_best:
        # Validate keys exist
        bad = [c for c in bold_best.keys() if c not in formatted.columns]
        if bad:
            raise KeyError(f"bold_best columns not found: {bad}")
        formatted = _bold_best_per_group(formatted, group_cols, list(bold_best.keys()),
                                         mode=None)  # we'll per-column below
        # Do per-column mode by iterating columns
        # (Re-apply because _bold_best_per_group handles one mode at a time.)
        for col, mode in bold_best.items():
            formatted = _bold_best_per_group(formatted, group_cols, [col], mode=mode)

    # 6) Escape text if requested
    if escape_text:
        def _esc_cell(v):
            # Don't escape macro-wrapped cells (e.g., \textbf{...})
            s = str(v)
            if s.startswith(r"\textbf{") and s.endswith("}"):
                inner = s[len(r"\textbf{"):-1]
                return rf"\textbf{{{escape_latex(inner)}}}"
            return escape_latex(s)
        formatted = formatted.map(_esc_cell) if hasattr(formatted, "map") else formatted.applymap(_esc_cell)


    # 7) Build LaTeX
    cols = list(formatted.columns)
    if colspec is None:
        colspec = "l" * len(cols)

    header = " & ".join(cols) + r" \\"
    body_rows = [" & ".join(map(str, row)) + r" \\"
                 for row in formatted.itertuples(index=False, name=None)]
    body = "\n".join(body_rows)

    prefix = r"\small" if use_small else ""
    begin_tabular = rf"\begin{{tabular}}{{{colspec}}}"
    end_tabular = r"\end{tabular}"

    tabular_block = rf"""{begin_tabular}
\toprule
{header}
\midrule
{body}
\bottomrule
{end_tabular}"""

    if resize_to_textwidth:
        tabular_block = rf"\resizebox{{\textwidth}}{{!}}{{{tabular_block}}}"

    note_block = ""
    if note:
        note_block = rf"""
\begin{{flushleft}}
\footnotesize {escape_latex(note) if escape_text else note}
\end{{flushleft}}"""

    tex = rf"""\begin{{table}}[{placement}]
\centering
{prefix}
\caption{{{escape_latex(caption) if escape_text else caption}}}
\label{{{label}}}
{tabular_block}
{note_block}
\end{{table}}"""
    return tex
