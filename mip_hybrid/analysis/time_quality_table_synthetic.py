import pandas as pd
import numpy as np
import glob
import os

# Find most recent CSV
csv_dir = r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid\MIP Hybrid Solver\experiments\rail"
csv_files = glob.glob(os.path.join(csv_dir, "syn_results_20251005_143600.csv"))  # FIXED: find any syn results
csv_path = max(csv_files, key=os.path.getmtime)

print(f"Using: {os.path.basename(csv_path)}\n")

df = pd.read_csv(csv_path)
small = df[(df['n'] == 200) & (df['m'] == 4000)]
large = df[(df['n'] == 300) & (df['m'] == 6000)]

print("| Time Budget | 200×4000 | 300×6000 | Winner |")
print("|-------------|----------|----------|--------|")

for t in [1, 5, 10, 15, 20, 25, 30]:
    small_hyb_objs = small['hyb_int'].dropna()
    large_hyb_objs = large['hyb_int'].dropna()
    
    small_mip_at_t = []
    large_mip_at_t = []
    
    # FIXED: Small loop now tracks BEST like large loop
    for _, row in small.iterrows():
        best_mip = None
        for i in range(100):
            if pd.isna(row.get(f'anytime_t_{i}', np.nan)):
                break
            if row[f'anytime_t_{i}'] <= t and 'gurobi' in str(row.get(f'anytime_method_{i}', '')).lower():
                obj = row[f'anytime_obj_{i}']
                if best_mip is None or obj < best_mip:
                    best_mip = obj
        if best_mip is not None:
            small_mip_at_t.append(best_mip)
    
    # Large loop (already correct, just removed debug)
    for _, row in large.iterrows():
        best_mip = None
        for i in range(100):
            if pd.isna(row.get(f'anytime_t_{i}', np.nan)):
                break
            if row[f'anytime_t_{i}'] <= t and 'gurobi' in str(row.get(f'anytime_method_{i}', '')).lower():
                obj = row[f'anytime_obj_{i}']
                if best_mip is None or obj < best_mip:
                    best_mip = obj
        if best_mip is not None:
            large_mip_at_t.append(best_mip)
    
    # Format and determine winner
    sh = f"HYB: {np.mean(small_hyb_objs):.1f}±{np.std(small_hyb_objs):.1f}" if len(small_hyb_objs) > 0 else "HYB: —"
    sm = f"MIP: {np.mean(small_mip_at_t):.1f}±{np.std(small_mip_at_t):.1f}" if len(small_mip_at_t) > 0 else "MIP: —"
    lh = f"HYB: {np.mean(large_hyb_objs):.1f}±{np.std(large_hyb_objs):.1f}" if len(large_hyb_objs) > 0 else "HYB: —"
    lm = f"MIP: {np.mean(large_mip_at_t):.1f}±{np.std(large_mip_at_t):.1f}" if len(large_mip_at_t) > 0 else "MIP: —"
    
    # Determine winner
    small_hyb_mean = np.mean(small_hyb_objs) if len(small_hyb_objs) > 0 else np.inf
    small_mip_mean = np.mean(small_mip_at_t) if len(small_mip_at_t) > 0 else np.inf
    large_hyb_mean = np.mean(large_hyb_objs) if len(large_hyb_objs) > 0 else np.inf
    large_mip_mean = np.mean(large_mip_at_t) if len(large_mip_at_t) > 0 else np.inf
    
    if small_hyb_mean < small_mip_mean and large_hyb_mean < large_mip_mean:
        winner = "**HYB**"
    elif small_mip_mean < small_hyb_mean and large_mip_mean < large_hyb_mean:
        winner = "**MIP**"
    else:
        winner = "Mixed"
    
    print(f"| {t} seconds | {sh}<br>{sm} | {lh}<br>{lm} | {winner} |")