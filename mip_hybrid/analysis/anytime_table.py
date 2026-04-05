# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 11:11:25 2025

@author: rich
"""

"""
Anytime Performance Analysis for MIP Hybrid vs Gurobi

Add this to your codebase as: mip_hybrid/analysis/anytime_table.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


def sample_anytime_at_checkpoints(anytime_log: List[Tuple[float, float, str]], 
                                   checkpoints: List[float]) -> Dict[float, float]:
    """
    Sample anytime log at specific time checkpoints.
    
    For each checkpoint, return the best objective value achieved by that time.
    
    Args:
        anytime_log: List of (time, objective, method) tuples
        checkpoints: List of time values to sample at (e.g., [1, 5, 10, 15, 20, 25, 30])
    
    Returns:
        Dictionary mapping checkpoint -> best objective value achieved by that time
    """
    if not anytime_log:
        return {cp: float('inf') for cp in checkpoints}
    
    # Sort by time
    sorted_log = sorted(anytime_log, key=lambda x: x[0])
    
    results = {}
    for checkpoint in checkpoints:
        # Find best objective achieved by this checkpoint
        best_obj = float('inf')
        for t, obj, method in sorted_log:
            if t <= checkpoint:
                best_obj = min(best_obj, obj)
            else:
                break  # No point checking further
        results[checkpoint] = best_obj
    
    return results


def aggregate_anytime_results(csv_path: str, 
                               checkpoints: List[float] = [1, 5, 10, 15, 20, 25, 30],
                               method_col: str = 'controller') -> pd.DataFrame:
    """
    Aggregate anytime performance across trials from a results CSV.
    
    Args:
        csv_path: Path to results CSV file
        checkpoints: Time checkpoints to evaluate at
        method_col: Column name indicating method (e.g., 'HYB', 'MIP')
    
    Returns:
        DataFrame with columns: method, checkpoint, mean, std, min, max, count
    """
    df = pd.read_csv(csv_path)
    
    records = []
    
    for method in df[method_col].unique():
        method_df = df[df[method_col] == method]
        
        for checkpoint in checkpoints:
            # Collect objective values at this checkpoint across all trials
            checkpoint_objs = []
            
            for idx, row in method_df.iterrows():
                # Parse anytime log for this trial
                anytime_log = []
                
                # Extract anytime data from columns (anytime_t_0, anytime_obj_0, etc.)
                i = 0
                while f'anytime_t_{i}' in row.index:
                    if pd.notna(row[f'anytime_t_{i}']):
                        t = row[f'anytime_t_{i}']
                        obj = row[f'anytime_obj_{i}']
                        method_name = row.get(f'anytime_method_{i}', 'unknown')
                        anytime_log.append((t, obj, method_name))
                    i += 1
                
                # Sample at checkpoint
                sampled = sample_anytime_at_checkpoints(anytime_log, [checkpoint])
                checkpoint_objs.append(sampled[checkpoint])
            
            # Filter out infinite values
            valid_objs = [obj for obj in checkpoint_objs if np.isfinite(obj)]
            
            if valid_objs:
                records.append({
                    'method': method,
                    'checkpoint_sec': checkpoint,
                    'mean': np.mean(valid_objs),
                    'std': np.std(valid_objs),
                    'min': np.min(valid_objs),
                    'max': np.max(valid_objs),
                    'count': len(valid_objs)
                })
    
    return pd.DataFrame(records)


def create_anytime_comparison_table(csv_path: str,
                                     checkpoints: List[float] = [1, 5, 10, 15, 20, 25, 30],
                                     output_format: str = 'latex') -> str:
    """
    Create a formatted comparison table showing anytime performance.
    
    Args:
        csv_path: Path to results CSV
        checkpoints: Time checkpoints to evaluate
        output_format: 'latex', 'markdown', or 'html'
    
    Returns:
        Formatted table as string
    """
    agg_df = aggregate_anytime_results(csv_path, checkpoints)
    
    # Pivot to get HYB and MIP side by side
    pivot_mean = agg_df.pivot(index='checkpoint_sec', columns='method', values='mean')
    pivot_std = agg_df.pivot(index='checkpoint_sec', columns='method', values='std')
    
    if output_format == 'latex':
        return _format_latex_table(pivot_mean, pivot_std, checkpoints)
    elif output_format == 'markdown':
        return _format_markdown_table(pivot_mean, pivot_std, checkpoints)
    elif output_format == 'html':
        return _format_html_table(pivot_mean, pivot_std, checkpoints)
    else:
        raise ValueError(f"Unknown format: {output_format}")


def _format_latex_table(pivot_mean: pd.DataFrame, 
                        pivot_std: pd.DataFrame,
                        checkpoints: List[float]) -> str:
    """Generate LaTeX table"""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Anytime Performance Comparison: Hybrid vs Gurobi}")
    lines.append(r"\label{tab:anytime}")
    
    # Determine columns present
    methods = [col for col in ['HYB', 'MIP', 'Gurobi'] if col in pivot_mean.columns]
    n_methods = len(methods)
    
    lines.append(r"\begin{tabular}{r" + "rr" * n_methods + "}")
    lines.append(r"\toprule")
    
    # Header
    header = "Time (s)"
    for method in methods:
        header += f" & {method} & $\\pm\\sigma$"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    for cp in checkpoints:
        if cp in pivot_mean.index:
            row = f"{cp:.0f}"
            for method in methods:
                mean_val = pivot_mean.loc[cp, method] if method in pivot_mean.columns else np.nan
                std_val = pivot_std.loc[cp, method] if method in pivot_std.columns else np.nan
                
                if np.isfinite(mean_val):
                    row += f" & {mean_val:.1f} & {std_val:.1f}"
                else:
                    row += " & --- & ---"
            row += r" \\"
            lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def _format_markdown_table(pivot_mean: pd.DataFrame,
                           pivot_std: pd.DataFrame,
                           checkpoints: List[float]) -> str:
    """Generate Markdown table"""
    methods = [col for col in ['HYB', 'MIP', 'Gurobi'] if col in pivot_mean.columns]
    
    lines = []
    lines.append("# Anytime Performance Comparison\n")
    
    # Header
    header = "| Time (s) |"
    separator = "|----------|"
    for method in methods:
        header += f" {method} (mean ± std) |"
        separator += ":-----------------:|"
    lines.append(header)
    lines.append(separator)
    
    # Data rows
    for cp in checkpoints:
        if cp in pivot_mean.index:
            row = f"| {cp:.0f} |"
            for method in methods:
                mean_val = pivot_mean.loc[cp, method] if method in pivot_mean.columns else np.nan
                std_val = pivot_std.loc[cp, method] if method in pivot_std.columns else np.nan
                
                if np.isfinite(mean_val):
                    row += f" {mean_val:.1f} ± {std_val:.1f} |"
                else:
                    row += " --- |"
            lines.append(row)
    
    return "\n".join(lines)


def _format_html_table(pivot_mean: pd.DataFrame,
                       pivot_std: pd.DataFrame,
                       checkpoints: List[float]) -> str:
    """Generate HTML table"""
    methods = [col for col in ['HYB', 'MIP', 'Gurobi'] if col in pivot_mean.columns]
    
    lines = []
    lines.append('<table border="1" style="border-collapse: collapse;">')
    lines.append('  <caption>Anytime Performance Comparison</caption>')
    lines.append('  <thead>')
    
    # Header
    header = '    <tr><th>Time (s)</th>'
    for method in methods:
        header += f'<th>{method} (mean ± std)</th>'
    header += '</tr>'
    lines.append(header)
    lines.append('  </thead>')
    lines.append('  <tbody>')
    
    # Data rows
    for cp in checkpoints:
        if cp in pivot_mean.index:
            row = f'    <tr><td>{cp:.0f}</td>'
            for method in methods:
                mean_val = pivot_mean.loc[cp, method] if method in pivot_mean.columns else np.nan
                std_val = pivot_std.loc[cp, method] if method in pivot_std.columns else np.nan
                
                if np.isfinite(mean_val):
                    row += f'<td>{mean_val:.1f} ± {std_val:.1f}</td>'
                else:
                    row += '<td>---</td>'
            row += '</tr>'
            lines.append(row)
    
    lines.append('  </tbody>')
    lines.append('</table>')
    
    return "\n".join(lines)


# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python anytime_table.py <results_csv> [output_format]")
        print("  output_format: latex (default), markdown, or html")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'latex'
    
    table = create_anytime_comparison_table(csv_path, output_format=output_format)
    print(table)