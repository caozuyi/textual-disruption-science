#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute year-level combination novelty trend
for Extended Data Fig. 2.

Input:
  - meta_table.parquet (publication-level)

Output:
  - Figure_combo_trend.csv (year-level, z-scored)

Notes:
  - Streaming over parquet row-groups (low memory)
  - No plotting
  - Fully reproducible
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import zscore

# ===============================
# CONFIG (EDIT PATHS IF NEEDED)
# ===============================
PARQUET_PATH = "data/meta_table.parquet"
OUT_DIR = "results/extended_data_fig2"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "Figure_combo_trend.csv")

YEAR = "sci_Year"
COMBO = "combo_novelty"

# ===============================
# LOAD PARQUET (STREAMING)
# ===============================
pf = pq.ParquetFile(PARQUET_PATH)
yearly_parts = []

for rg in range(pf.num_row_groups):
    df = pf.read_row_group(rg, columns=[YEAR, COMBO]).to_pandas()

    df[YEAR] = pd.to_numeric(df[YEAR], errors="coerce")
    df[COMBO] = pd.to_numeric(df[COMBO], errors="coerce")
    df = df.dropna(subset=[YEAR, COMBO])

    yearly = df.groupby(YEAR, as_index=False)[COMBO].mean()
    yearly_parts.append(yearly)

# ===============================
# AGGREGATE ACROSS ROW GROUPS
# ===============================
trend_df = (
    pd.concat(yearly_parts, ignore_index=True)
      .groupby(YEAR, as_index=False)[COMBO]
      .mean()
)

# Z-score normalization (as in paper)
trend_df[COMBO] = zscore(trend_df[COMBO], nan_policy="omit")

trend_df.to_csv(OUT_CSV, index=False)

print("✔ Extended Data Fig. 2 data saved to:")
print(" ", OUT_CSV)
