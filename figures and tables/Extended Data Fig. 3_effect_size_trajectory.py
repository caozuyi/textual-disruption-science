# =====================================================
# Extended Data Fig. 3 | Rolling-window estimates of the
# effect of textual disruption on citation outcomes
#
# LOW-MEMORY SAFE VERSION (GitHub / external review ready)
# =====================================================

"""
This script estimates rolling-window regression coefficients of textual
disruption on long-run citation recognition. The analysis is conducted
using symmetric rolling windows of ±5, ±10, and ±15 years to assess
the temporal robustness of the estimated effects.

This figure corresponds to Extended Data Fig. 3 in the manuscript.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

# Path to the merged meta-table (Parquet format)
PARQUET_PATH = Path("data/meta_table.parquet")   # user-provided

# Output directory
OUTDIR = Path("results/extended_data/fig3")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
YEAR_MIN, YEAR_MAX = 1900, 2020
ROLLING_WINDOWS = [5, 10, 15]   # symmetric ± window size
MIN_N = 5000                   # minimum observations per window

# =========================
# STEP 1: STREAM LOAD DATA
# =========================

print(">>> Streaming parquet row-groups (low-memory mode)...")
pf = pq.ParquetFile(PARQUET_PATH)

chunks = []

for rg in range(pf.num_row_groups):
    df = pf.read_row_group(
        rg,
        columns=[
            "sci_Year",
            "sci_Citation_Count",
            "Z_novelty",
            "Z_consolidation",
            "combo_novelty",
        ]
    ).to_pandas()

    # ---- Convert to numeric (robust to string storage) ----
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Construct textual disruption ----
    df["textual_disruption"] = df["Z_novelty"] - df["Z_consolidation"]

    # ---- Restrict analysis period ----
    df = df[df["sci_Year"].between(YEAR_MIN, YEAR_MAX)]

    # ---- Retain only required columns ----
    df = df[
        [
            "sci_Year",
            "sci_Citation_Count",
            "textual_disruption",
            "combo_novelty",
        ]
    ]

    chunks.append(df)

# Merge streamed chunks
df = pd.concat(chunks, ignore_index=True)
del chunks

print(f">>> Loaded {len(df):,} observations")

# =========================
# STEP 2: ROLLING REGRESSIONS
# =========================

records = []
print(">>> Running rolling-window regressions...")

for w in ROLLING_WINDOWS:
    print(f"--- Symmetric rolling window: ±{w} years ---")

    for center_year in range(YEAR_MIN + w, YEAR_MAX - w):
        sub = df[
            (df["sci_Year"] >= center_year - w)
            & (df["sci_Year"] <= center_year + w)
        ].dropna()

        if len(sub) < MIN_N:
            continue

        X = sm.add_constant(
            sub[["textual_disruption", "combo_novelty"]],
            has_constant="add",
        )
        y = np.log1p(sub["sci_Citation_Count"])

        try:
            res = sm.OLS(y, X).fit(cov_type="HC3")
            records.append(
                {
                    "center_year": center_year,
                    "window": w,
                    "beta": res.params["textual_disruption"],
                    "se": res.bse["textual_disruption"],
                    "n": len(sub),
                }
            )
        except Exception:
            continue

# =========================
# STEP 3: SAVE RESULTS
# =========================

results_df = pd.DataFrame(records)
csv_path = OUTDIR / "extdata_fig3_rolling_effects.csv"
results_df.to_csv(csv_path, index=False)

print(f">>> Saved results to {csv_path}")

# =========================
# STEP 4: PLOT EXTENDED DATA FIGURE
# =========================

plt.figure(figsize=(9, 5))

for w in ROLLING_WINDOWS:
    sub = results_df[results_df["window"] == w]
    plt.plot(
        sub["center_year"],
        sub["beta"],
        label=f"±{w}-year window",
        linewidth=2,
    )

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Center year of rolling window")
plt.ylabel("Estimated effect of textual disruption (β)")
plt.legend(frameon=False)
plt.tight_layout()

fig_path = OUTDIR / "extdata_fig3_rolling_window_effects.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f">>> Extended Data Fig. 3 saved to {fig_path}")
print(">>> DONE")
