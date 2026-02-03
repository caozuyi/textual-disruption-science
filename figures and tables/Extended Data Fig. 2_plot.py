#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Extended Data Fig. 2:
Year-level trajectory of combination novelty.

Input:
  - Figure_combo_trend.csv

Output:
  - ExtendedDataFig2_combo_novelty_trend.png

Notes:
  - Visualization only
  - 5-year rolling mean (Nature-acceptable smoothing)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_CSV = "results/extended_data_fig2/Figure_combo_trend.csv"
OUT_FIG  = "results/extended_data_fig2/ExtendedDataFig2_combo_novelty_trend.png"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_CSV)
df = df.sort_values("sci_Year")

# =========================
# SMOOTHING (ROLLING MEAN)
# =========================
df["combo_novelty_5yr"] = (
    df["combo_novelty"]
    .rolling(window=5, center=True, min_periods=1)
    .mean()
)

# =========================
# PLOT (NATURE STYLE)
# =========================
plt.figure(figsize=(6.5, 3.8))

plt.plot(
    df["sci_Year"],
    df["combo_novelty"],
    color="black",
    alpha=0.25,
    linewidth=1,
    label="Year-level"
)

plt.plot(
    df["sci_Year"],
    df["combo_novelty_5yr"],
    color="black",
    linewidth=2,
    label="5-year rolling mean"
)

plt.axhline(0, color="black", linestyle="--", linewidth=0.8)

plt.xlabel("Year", fontsize=11)
plt.ylabel("Combination novelty", fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(frameon=False, fontsize=9, loc="upper right")
plt.tight_layout()

plt.savefig(OUT_FIG, dpi=600)
plt.close()

print("✔ Extended Data Fig. 2 figure saved to:")
print(" ", OUT_FIG)
