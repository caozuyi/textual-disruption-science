#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2 (Panels a & b): Yearly Aggregation of Textual and Citation-based Indicators
===============================================================================

This script generates the yearly aggregated measures underlying Figure 2a and Figure 2b
in the manuscript.

- Figure 2a: Long-term temporal trends in standardized textual and citation-based indicators
- Figure 2b: Period-wise comparison based on the same yearly aggregates

The implementation mirrors the original large-scale computation logic used in the study.
Due to the scale of the full dataset, production runs were executed on a distributed
high-performance computing environment. This public version documents the analytical logic
in a transparent and inspectable form.

Note:
- Figure 2b does not rely on additional raw data or alternative computations.
- It is constructed by re-aggregating and normalizing the yearly outputs produced here.
"""

import os
import csv
from collections import defaultdict

import pyarrow.dataset as ds


# =====================================================
# Configuration
# =====================================================

# Path to the paper-level dataset (parquet format)
PARQUET_PATH = "data/meta_table.parquet"

# Output directory
OUT_DIR = "output/figure2"
os.makedirs(OUT_DIR, exist_ok=True)

# Output file
OUT_CSV = os.path.join(OUT_DIR, "figure2_yearly_aggregates.csv")

# Columns required for Figure 2
COLS = [
    "sci_Year",
    "novelty_raw",
    "consolidation_raw",
    "Z_novelty",
    "Z_consolidation",
    "textual_disruption",
    "sci_Disruption",
]


# =====================================================
# Dataset initialization
# =====================================================

dataset = ds.dataset(PARQUET_PATH, format="parquet")

# Accumulators for yearly aggregation
sum_dict = defaultdict(lambda: defaultdict(float))
count_dict = defaultdict(int)


# =====================================================
# Batch-wise scan (HPC-safe logic)
# =====================================================

scanner = dataset.scanner(
    columns=COLS,
    batch_size=200_000
)

for batch in scanner.to_batches():
    cols = batch.to_pydict()
    years = cols["sci_Year"]

    for i, y in enumerate(years):
        if y is None:
            continue

        year = int(float(y))
        count_dict[year] += 1

        for v in [
            "novelty_raw",
            "consolidation_raw",
            "Z_novelty",
            "Z_consolidation",
            "textual_disruption",
            "sci_Disruption",
        ]:
            val = cols[v][i]
            if val is not None:
                sum_dict[year][v] += float(val)


# =====================================================
# Write yearly means to CSV
# =====================================================

years_sorted = sorted(count_dict.keys())

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "year",
        "novelty_raw",
        "consolidation_raw",
        "Z_novelty",
        "Z_consolidation",
        "textual_disruption",
        "sci_Disruption",
    ])

    for y in years_sorted:
        n = count_dict[y]
        writer.writerow([
            y,
            sum_dict[y]["novelty_raw"] / n if n else None,
            sum_dict[y]["consolidation_raw"] / n if n else None,
            sum_dict[y]["Z_novelty"] / n if n else None,
            sum_dict[y]["Z_consolidation"] / n if n else None,
            sum_dict[y]["textual_disruption"] / n if n else None,
            sum_dict[y]["sci_Disruption"] / n if n else None,
        ])

print("SUCCESS: Figure 2 yearly aggregates written to:")
print(OUT_CSV)
