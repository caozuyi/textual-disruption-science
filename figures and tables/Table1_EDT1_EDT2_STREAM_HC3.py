#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================
Table 1 and Extended Data Tables (Streaming OLS + HC3)
=========================================================

This script generates:
- Table 1 (Main Text): Long-run citation impact (log(1 + sci_C10))
- Extended Data Table 1: Alternative citation windows (log(1 + sci_Citation_Count))
- Extended Data Table 2: Disruption outcomes (sci_Disruption)

All models are estimated using:
- Streaming OLS over a large Parquet meta-table
- HC3 heteroskedasticity-robust standard errors
- Year-centered interaction terms
- Log-transformed control variables

The implementation is designed to be memory-safe for
datasets with tens of millions of observations.

Author: 
"""

# =========================================================
# Imports
# =========================================================

import os
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

# =========================================================
# Paths (USER SHOULD MODIFY)
# =========================================================

PARQUET_PATH = "meta_table.parquet"   # unified meta-table
OUT_DIR = "tables_output"
os.makedirs(OUT_DIR, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_FILE = os.path.join(
    OUT_DIR,
    f"table1_and_extended_tables_streaming_hc3_{STAMP}.csv"
)

# =========================================================
# Dependent variables (paper mapping)
# =========================================================

DEP_MAP = {
    "Table1_logC10":        "sci_C10",
    "ED_Table1_logCitation":"sci_Citation_Count",
    "ED_Table2_Disruption":"sci_Disruption",
}

# =========================================================
# Variable groups (match paper)
# =========================================================

YEAR_COL = "sci_Year"

GEN_VARS  = ["Z_novelty", "Z_consolidation"]
PERF_VARS = ["textual_disruption", "combo_novelty"]

CTRL_RAW = [
    "sci_Team_Size",
    "sci_Institution_Count",
    "sci_Reference_Count",
]

CTRL_LOG = ["ctrl_team", "ctrl_inst", "ctrl_refs"]

READ_COLS = list(set(
    [YEAR_COL]
    + GEN_VARS
    + PERF_VARS
    + CTRL_RAW
    + list(DEP_MAP.values())
))

# =========================================================
# Utility functions
# =========================================================

def safe_numeric(series):
    """Convert to numeric safely."""
    return pd.to_numeric(series, errors="coerce")

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def two_sided_pvalue(t):
    return 2.0 * (1.0 - normal_cdf(abs(t)))

# =========================================================
# Core function: Streaming OLS with HC3
# =========================================================

def streaming_ols_hc3(parquet_path, dep_var, x_vars, model_label):
    """
    Estimate OLS with HC3 standard errors using streaming Parquet reads.

    Parameters
    ----------
    parquet_path : str
        Path to the meta-table Parquet file.
    dep_var : str
        Dependent variable column name.
    x_vars : list
        List of main explanatory variables.
    model_label : str
        Label for model identification.

    Returns
    -------
    list of dict
        Regression results in tidy format.
    """

    pf = pq.ParquetFile(parquet_path)
    stats = None
    x_names_ref = None

    # =========================
    # PASS 0: Mean & Std
    # =========================
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=READ_COLS).to_pandas()
        for c in df.columns:
            df[c] = safe_numeric(df[c])

        df = df[df[YEAR_COL].between(1900, 2021)]
        if df.empty:
            continue

        df["Year_c"] = df[YEAR_COL] - 1980.0
        df["ctrl_team"] = np.log1p(df["sci_Team_Size"])
        df["ctrl_inst"] = np.log1p(df["sci_Institution_Count"])
        df["ctrl_refs"] = np.log1p(df["sci_Reference_Count"])

        if dep_var == "sci_Disruption":
            df["Y"] = df[dep_var]
        else:
            df["Y"] = np.log1p(df[dep_var])

        for v in x_vars:
            df[f"{v}_x_year"] = df[v] * df["Year_c"]

        x_names = (
            x_vars +
            ["Year_c"] +
            CTRL_LOG +
            [f"{v}_x_year" for v in x_vars]
        )

        dft = df[["Y"] + x_names].dropna()
        if dft.empty:
            continue

        M = dft.to_numpy(dtype=np.float64)

        if stats is None:
            stats = {
                "n": 0,
                "sum": np.zeros(M.shape[1]),
                "sumsq": np.zeros(M.shape[1]),
            }
            x_names_ref = x_names

        stats["n"] += M.shape[0]
        stats["sum"] += M.sum(axis=0)
        stats["sumsq"] += (M * M).sum(axis=0)

    mean = stats["sum"] / stats["n"]
    var = stats["sumsq"] / stats["n"] - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))

    y_mean, y_std = mean[0], std[0]
    X_mean, X_std = mean[1:], std[1:]
    N = stats["n"]

    # =========================
    # PASS 1: OLS
    # =========================
    k = 1 + len(x_names_ref)
    XtX = np.zeros((k, k))
    Xty = np.zeros(k)
    sumy = 0.0
    sumy2 = 0.0

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=READ_COLS).to_pandas()
        for c in df.columns:
            df[c] = safe_numeric(df[c])

        df = df[df[YEAR_COL].between(1900, 2021)]
        if df.empty:
            continue

        df["Year_c"] = df[YEAR_COL] - 1980.0
        df["ctrl_team"] = np.log1p(df["sci_Team_Size"])
        df["ctrl_inst"] = np.log1p(df["sci_Institution_Count"])
        df["ctrl_refs"] = np.log1p(df["sci_Reference_Count"])

        if dep_var == "sci_Disruption":
            df["Y"] = df[dep_var]
        else:
            df["Y"] = np.log1p(df[dep_var])

        for v in x_vars:
            df[f"{v}_x_year"] = df[v] * df["Year_c"]

        dft = df[["Y"] + x_names_ref].dropna()
        if dft.empty:
            continue

        y = dft["Y"].to_numpy()
        Xraw = dft[x_names_ref].to_numpy()

        yZ = (y - y_mean) / y_std
        XZ = (Xraw - X_mean) / X_std
        X = np.hstack([np.ones((XZ.shape[0], 1)), XZ])

        XtX += X.T @ X
        Xty += X.T @ yZ
        sumy += yZ.sum()
        sumy2 += (yZ * yZ).sum()

    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ Xty

    # =========================
    # PASS 2: HC3
    # =========================
    meat = np.zeros((k, k))
    SSE = 0.0
    ybar = sumy / N
    TSS = sumy2 - N * ybar * ybar

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=READ_COLS).to_pandas()
        for c in df.columns:
            df[c] = safe_numeric(df[c])

        df = df[df[YEAR_COL].between(1900, 2021)]
        if df.empty:
            continue

        df["Year_c"] = df[YEAR_COL] - 1980.0
        df["ctrl_team"] = np.log1p(df["sci_Team_Size"])
        df["ctrl_inst"] = np.log1p(df["sci_Institution_Count"])
        df["ctrl_refs"] = np.log1p(df["sci_Reference_Count"])

        if dep_var == "sci_Disruption":
            df["Y"] = df[dep_var]
        else:
            df["Y"] = np.log1p(df[dep_var])

        for v in x_vars:
            df[f"{v}_x_year"] = df[v] * df["Year_c"]

        dft = df[["Y"] + x_names_ref].dropna()
        if dft.empty:
            continue

        y = dft["Y"].to_numpy()
        Xraw = dft[x_names_ref].to_numpy()

        yZ = (y - y_mean) / y_std
        XZ = (Xraw - X_mean) / X_std
        X = np.hstack([np.ones((XZ.shape[0], 1)), XZ])

        e = yZ - X @ beta
        SSE += (e * e).sum()

        H = X @ XtX_inv
        h = np.sum(H * X, axis=1)
        w = (e * e) / np.square(1.0 - h)
        Xw = X * np.sqrt(np.nan_to_num(w))[:, None]
        meat += Xw.T @ Xw

    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = np.array([two_sided_pvalue(t) for t in tvals])
    R2 = 1.0 - SSE / TSS

    rows = []
    names = ["Intercept"] + x_names_ref
    for i, name in enumerate(names):
        rows.append({
            "Dependent": dep_var,
            "Model": model_label,
            "Variable": name,
            "Coef": float(beta[i]),
            "SE": float(se[i]),
            "t": float(tvals[i]),
            "p": float(pvals[i]),
            "N": int(N),
            "R2": float(R2),
        })

    return rows

# =========================================================
# Run all models
# =========================================================

all_results = []

for label, dep in DEP_MAP.items():
    all_results += streaming_ols_hc3(
        PARQUET_PATH, dep, GEN_VARS, "Generative"
    )
    all_results += streaming_ols_hc3(
        PARQUET_PATH, dep, PERF_VARS, "Performative"
    )

df_out = pd.DataFrame(all_results)
df_out.to_csv(OUT_FILE, index=False)

print("✓ All tables generated:", OUT_FILE)

