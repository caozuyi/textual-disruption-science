#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3 | Period-specific citation rewards of textual innovation

This script generates both panels (a) and (b) of Figure 3 in the manuscript:
"When disruptive science does not decline: Textual innovation and the decoupling
of idea generation and scientific recognition, 1900–2021".

Analysis:
- Period-specific OLS regressions
- Dependent variable: log(1 + sci_Citation_Count)
- Robust standard errors (HC3)
- Output: regression table (CSV) + two-panel forest plot (PNG)

Data requirement:
- meta_table.parquet (paper-level merged dataset)

Author: 
"""

# =========================================================
# 0. Imports
# =========================================================

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# 1. Paths (EDIT AS NEEDED)
# =========================================================

DATA_PATH = "data/meta_table.parquet"
OUT_CSV   = "outputs/figure3_period_results_citation_all.csv"
OUT_PNG   = "outputs/figure3_forestplot_citation_all.png"

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# =========================================================
# 2. Load required columns only
# =========================================================

COLS = [
    "sci_Year",
    "sci_Citation_Count",
    "Z_novelty",
    "Z_consolidation",
    "textual_disruption",
    "combo_novelty",
    "sci_Team_Size",
    "sci_Institution_Count",
    "sci_Reference_Count",
]

print("Loading data...")
df = pd.read_parquet(DATA_PATH, columns=COLS)
print(f"Rows loaded: {len(df):,}")

# =========================================================
# 3. Type conversion
# =========================================================

for c in COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================================================
# 4. Dependent variable & controls
# =========================================================

df["DV"] = np.log1p(df["sci_Citation_Count"])
df["log_team"] = np.log1p(df["sci_Team_Size"])
df["log_inst"] = np.log1p(df["sci_Institution_Count"])
df["log_refs"] = np.log1p(df["sci_Reference_Count"])

TEXTUAL_VARS = [
    "Z_novelty",
    "Z_consolidation",
    "textual_disruption",
    "combo_novelty",
]

CONTROLS = ["log_team", "log_inst", "log_refs"]

# =========================================================
# 5. Period definitions
# =========================================================

PERIODS = {
    "1900–1945": (1900, 1945),
    "1946–1980": (1946, 1980),
    "1981–2000": (1981, 2000),
    "2001–2021": (2001, 2021),
}

# =========================================================
# 6. Run period-specific regressions
# =========================================================

print("Running period-specific regressions...")
results = []

for label, (start, end) in PERIODS.items():

    sub = df[
        (df["sci_Year"] >= start) &
        (df["sci_Year"] <= end)
    ][["DV"] + TEXTUAL_VARS + CONTROLS].dropna()

    print(f"{label}: N = {len(sub):,}")

    for var in TEXTUAL_VARS:
        X = sm.add_constant(sub[[var] + CONTROLS])
        y = sub["DV"]

        model = sm.OLS(y, X).fit(cov_type="HC3")

        results.append({
            "Period": label,
            "Variable": var,
            "Beta": model.params[var],
            "SE": model.bse[var],
            "CI_low": model.conf_int().loc[var, 0],
            "CI_high": model.conf_int().loc[var, 1],
            "N": int(model.nobs),
        })

# =========================================================
# 7. Save regression results
# =========================================================

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_CSV, index=False)
print("Saved regression table:", OUT_CSV)

# =========================================================
# 8. Plot Figure 3 (forest plot)
# =========================================================

period_order = ["1900–1945", "1946–1980", "1981–2000", "2001–2021"]
res_df["Period"] = pd.Categorical(
    res_df["Period"], categories=period_order, ordered=True
)

LEFT_VARS  = ["Z_novelty", "Z_consolidation"]
RIGHT_VARS = ["textual_disruption", "combo_novelty"]

VAR_LABELS = {
    "Z_novelty": "Z-novelty",
    "Z_consolidation": "Z-consolidation",
    "textual_disruption": "Textual disruption",
    "combo_novelty": "Combinational novelty",
}

def forest_panel(ax, data, variables):
    data = data[data["Variable"].isin(variables)].copy()
    data = data.sort_values("Period")

    for var in variables:
        sub = data[data["Variable"] == var]
        y = sub["Period"].cat.codes.values
        coef = sub["Beta"].values
        se = sub["SE"].values

        ax.errorbar(
            coef,
            y,
            xerr=1.96 * se,
            fmt="o",
            capsize=4,
            label=VAR_LABELS[var]
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(period_order)))
    ax.set_yticklabels(period_order)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

forest_panel(axes[0], res_df, LEFT_VARS)
forest_panel(axes[1], res_df, RIGHT_VARS)

axes[0].set_title("a")
axes[1].set_title("b")

fig.suptitle(
    "Figure 3 | Period-specific citation rewards of textual innovation",
    fontsize=13
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print("Saved figure:", OUT_PNG)
print("Figure 3 pipeline completed successfully.")

