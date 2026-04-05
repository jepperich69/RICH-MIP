import pandas as pd
import numpy as np
import glob
import os

csv_dir = r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid\MIP Hybrid Solver\experiments\rail"
#csv_files = glob.glob(os.path.join(csv_dir, "rail582_results_20251005_181134.csv"))
csv_files = glob.glob(os.path.join(csv_dir, "rail582_results_20251005_203022.csv"))

csv_path = max(csv_files, key=os.path.getmtime)

print(f"Using: {os.path.basename(csv_path)}\n")
df = pd.read_csv(csv_path)

# Debug: check one row
row = df.iloc[0]
print("DEBUG - First row anytime data:")
for i in range(14):
    if pd.notna(row.get(f'anytime_method_{i}')):
        print(f"  [{i}] {row[f'anytime_method_{i}']}: obj={row[f'anytime_obj_{i}']:.0f} at t={row[f'anytime_t_{i}']:.3f}s")

print(f"\nhyb_total: {row['hyb_total']:.2f}s")
print(f"polish_obj: {row['polish_obj']:.0f}\n")

print("Rail582 Time-Quality Analysis")
print("\n| Time Budget | Hybrid | MIP (Gurobi) | Winner |")
print("|-------------|--------|--------------|--------|")

for t in [0.5, 1.0, 1.5, 2.0, 3.0]:
    hyb_at_t = []
    mip_at_t = []
    
    for _, row in df.iterrows():
        # Hybrid: collect all non-MIP solutions by time t
        best_hyb = None
        for i in range(100):
            method = str(row.get(f'anytime_method_{i}', ''))
            if method == '' or method == 'nan':
                break
            obj = row.get(f'anytime_obj_{i}', np.nan)
            time_i = row.get(f'anytime_t_{i}', np.inf)
            
            # Skip if this is MIP or obj is NaN
            if 'gurobi' in method.lower() or pd.isna(obj):
                continue
                
            # Take best hybrid solution by time t
            if time_i <= t:
                if best_hyb is None or obj < best_hyb:
                    best_hyb = obj
        
        if best_hyb is not None:
            hyb_at_t.append(best_hyb)
        
        # MIP: best by time t
        best_mip = None
        for i in range(100):
            method = str(row.get(f'anytime_method_{i}', ''))
            if method == '' or method == 'nan':
                break
            if 'gurobi' in method.lower():
                time_i = row.get(f'anytime_t_{i}', np.inf)
                if time_i <= t:
                    obj = row[f'anytime_obj_{i}']
                    if best_mip is None or obj < best_mip:
                        best_mip = obj
        if best_mip is not None:
            mip_at_t.append(best_mip)
    
    hyb_str = f"{np.mean(hyb_at_t):.1f}±{np.std(hyb_at_t):.1f}" if len(hyb_at_t) > 0 else "—"
    mip_str = f"{np.mean(mip_at_t):.1f}±{np.std(mip_at_t):.1f}" if len(mip_at_t) > 0 else "—"
    
    if len(hyb_at_t) > 0 and len(mip_at_t) > 0:
        if np.mean(hyb_at_t) < np.mean(mip_at_t) * 0.99:
            winner = "**Hybrid**"
        elif np.mean(mip_at_t) < np.mean(hyb_at_t) * 0.99:
            winner = "**MIP**"
        else:
            winner = "Tie"
    else:
        winner = "—"
    
    print(f"| {t}s | {hyb_str} | {mip_str} | {winner} |")

print(f"\n### Final (no time limit)")
print(f"Hybrid: {df['polish_obj'].mean():.1f}±{df['polish_obj'].std():.1f} in {df['hyb_total'].mean():.2f}s")
print(f"MIP: {df['mip_obj'].mean():.1f} in {df['mip_time'].mean():.2f}s")


###########

import pandas as pd

#df = pd.read_csv("rail582_results_20251005_182041.csv")  # Use the actual filename

print(f"Total trials: {len(df)}")
print(f"\nPolish objective distribution:")
print(df['polish_obj'].value_counts().sort_index())
print(f"\nMean: {df['polish_obj'].mean():.1f}")
print(f"Std: {df['polish_obj'].std():.1f}")
print(f"Median: {df['polish_obj'].median():.1f}")

# Count how many achieve 213
count_213 = (df['polish_obj'] == 213).sum()
print(f"\nTrials achieving 213: {count_213}/{len(df)} ({100*count_213/len(df):.1f}%)")