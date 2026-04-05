# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 16:47:18 2025

@author: rich
"""

from mip_hybrid.analysis.anytime_table import table_gap_time_compare

# Point to your CSV
csv_path = r"experiments/rail/rail582_results_20251005_120836.csv"

# Generate the table
table = table_gap_time_compare(csv_path, output_format='markdown')
print(table)

# Save to file
with open('experiments/rail/table_gap_time_compare.md', 'w') as f:
    f.write(table)