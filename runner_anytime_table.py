import sys
sys.path.insert(0, r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid\MIP Hybrid Solver")

from mip_hybrid.analysis.anytime_table import create_anytime_comparison_table

# Generate the table
csv_path = r"experiments/rail/rail582_results_20251005_120836.csv"
table = create_anytime_comparison_table(csv_path, output_format='markdown')
print(table)


###

import pandas as pd

csv_path = r"experiments/rail/rail582_results_20251005_125343.csv"
df = pd.read_csv(csv_path)

# Check for all anytime method columns
for col in df.columns:
    if 'anytime_method' in col:
        print(f"{col}: {df[col].unique()}")

print(f"\nTotal anytime columns: {len([c for c in df.columns if 'anytime' in c])}")


###

# Check what's in the mip_res
csv_path = r"experiments/rail/rail582_results_20251005_125343.csv"
df = pd.read_csv(csv_path)

# Check if there's any indication of what solve_mip returned
print("Columns in CSV:")
print([c for c in df.columns if 'mip' in c.lower()])

# Check the anytime_summary to see what's there
print("\nAnytime summary:")
print(df['anytime_summary'].iloc[0])