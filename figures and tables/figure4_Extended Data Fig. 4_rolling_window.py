#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4 & Extended Data Fig. 4
Rolling-window regressions of textual disruption on citation recognition

This script generates:
- Main text Figure 4 (long-term citation horizon, sci_C10)
- Extended Data Fig. 4 (short-term citation horizon, sci_C5)

Method:
- Rolling-window OLS regressions
- Dependent variable: log(1 + citation count)
- Key explanatory variable: textual_disruption
- Windows: 5, 10, 15 years

Note:
This script operates on pre-aggregated yearly data derived from the
meta_table constructed by merging OpenAlex and SciSciNet records.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =====================================================
# Configuration
# =====================================================

# Directory containing yearly aggregated CSV files
DATA_DIR = "./data"

# Output directory for figures
OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Citation horizons
FILES = {
    "C10": ("Figure4_yearly_aggregates_C10_FROM_META.csv", "sci_C10"),
    "C5":  ("Figure4_yearly_aggregates_C5_FROM_META.csv",  "sci_C5"),
}

# Rolling-window sizes (years)
WINDOWS = [5, 10, 15]

# Key explanatory variable
X_VAR = "textual_disruption"

# Minimum total sample size within a window
MIN_N = 1000


# =====================================================
# Rolling regression function
# =====================================================

def rolling_regression(csv_file, dep_var, output_name):
    """
    Run rolling-window OLS regressions and plot coefficients.

    Parameters
    ----------
    csv_file : str
        Yearly aggregated data file.
    dep_var : str
        Citation variable (e.g., sci_C10 or sci_C5).
    output_name : str
        Output figure filename.
    """

    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    df = df.sort_values("Year").reset_index(drop=True)

    results = {}

    for w in WINDOWS:
        xs, ys = [], []

        for i in range(len(df) - w + 1):
            sub = df.iloc[i:i + w]

            # Ensure sufficient observations
            if sub["N"].sum() < MIN_N:
                continue

            y = np.log1p(sub[dep_var])
            X = sm.add_constant(sub[[X_VAR]])

            try:
                coef = sm.OLS(y, X).fit().params[X_VAR]
            except Exception:
                continue

            xs.append(sub["Year"].iloc[0])
            ys.append(coef)

        results[w] = (xs, ys)

    # =================================================
    # Plot
    # =================================================

    plt.figure(figsize=(12, 5))

    for w, (x, y) in results.items():
        plt.plot(x, y, label=f"{w}-year window", linewidth=2)

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Start year of rolling window")
    plt.ylabel(f"Rolling coefficient (log(1 + {dep_var}))")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, output_name)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


# =====================================================
# Run analyses
# =====================================================

# Main text Figure 4: long-term citation recognition
rolling_regression(
    csv_file=FILES["C10"][0],
    dep_var=FILES["C10"][1],
    output_name="Figure4_rolling_logC10_textual_disruption.png"
)

# Extended Data Fig. 4: short-term citation recognition
rolling_regression(
    csv_file=FILES["C5"][0],
    dep_var=FILES["C5"][1],
    output_name="ExtendedDataFig4_rolling_logC5_textual_disruption.png"
)

print("All rolling-window analyses completed.")
