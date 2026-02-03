#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Textual Innovation Indicators (Public Analysis Version)
=======================================================

This script implements the core textual innovation indicators reported in the manuscript:

1. Textual Novelty
2. Textual Consolidation
3. Textual Disruption
4. Combinational Novelty (Uzzi-style z-score)

The implementation mirrors the original large-scale computation logic used in the study.
Due to the scale of the full datasets, production runs were executed on a distributed
high-performance computing environment. This public version documents the analytical
logic in a transparent and inspectable form.

This script is intended for methodological inspection and demonstration.
"""

import pandas as pd
import numpy as np
import os

# ==========================================================
# Part 0. Utility functions
# ==========================================================

def zscore(series: pd.Series) -> pd.Series:
    """Standard z-score transformation."""
    return (series - series.mean()) / series.std()


# ==========================================================
# Part 1. Textual novelty
# ==========================================================

def compute_textual_novelty(df: pd.DataFrame) -> pd.Series:
    """
    Ex ante textual novelty: introduction of new linguistic material.
    """
    return (
        df["new_word"]
        + df["new_phrase"]
        + df["new_word_comb"]
        + df["new_phrase_comb"]
        + df["semantic_distance"]
    )


# ==========================================================
# Part 2. Textual consolidation
# ==========================================================

def compute_textual_consolidation(df: pd.DataFrame) -> pd.Series:
    """
    Ex post textual consolidation: subsequent reuse of introduced elements.
    """
    return (
        df["new_word_reuse"]
        + df["new_phrase_reuse"]
        + df["new_word_comb_reuse"]
        + df["new_phrase_comb_reuse"]
    )


# ==========================================================
# Part 3. Textual disruption
# ==========================================================

def compute_textual_disruption(
    novelty_raw: pd.Series,
    consolidation_raw: pd.Series
) -> pd.Series:
    """
    Textual disruption defined as the standardized divergence
    between novelty and consolidation.
    """
    z_novelty = zscore(novelty_raw)
    z_consolidation = zscore(consolidation_raw)
    return z_novelty - z_consolidation


# ==========================================================
# Part 4. Combinational novelty (Uzzi-style, chunked)
# ==========================================================

def compute_combinational_novelty_uzzi(
    word_freq: dict,
    path_word_combs: str,
    chunk_size: int = 2_000_000
) -> pd.DataFrame:
    """
    Compute Uzzi-style combinational novelty at the paper level.

    This implementation strictly follows the original two-stage aggregation logic:
    (1) aggregation within chunks
    (2) aggregation across chunks
    """

    paper_uzzi_parts = []
    chunk_id = 0

    for chunk in pd.read_csv(path_word_combs, chunksize=chunk_size):
        chunk_id += 1
        print(f"Processing combination chunk {chunk_id}...")

        # Marginal frequencies
        f1 = chunk["Word1"].map(word_freq).fillna(1.0).astype(float)
        f2 = chunk["Word2"].map(word_freq).fillna(1.0).astype(float)

        # Observed pair frequency (Reuse)
        f_pair = chunk["Reuse"].astype(float)

        # Expected frequency
        expected = np.sqrt(f1 * f2)

        # Uzzi-style z-score
        z = (f_pair - expected) / (expected + 1e-6)

        chunk["zscore"] = z

        # First aggregation: within chunk
        paper_z = (
            chunk
            .groupby("PaperID")["zscore"]
            .mean()
            .reset_index()
        )

        paper_uzzi_parts.append(paper_z)

    # Second aggregation: across chunks
    paper_uzzi = (
        pd.concat(paper_uzzi_parts, ignore_index=True)
        .groupby("PaperID")["zscore"]
        .mean()
        .reset_index()
        .rename(columns={"zscore": "combo_novelty"})
    )

    return paper_uzzi


# ==========================================================
# Part 5. End-to-end demonstration pipeline
# ==========================================================

def run_pipeline(
    metrics_df: pd.DataFrame,
    word_freq_df: pd.DataFrame,
    path_word_combs: str
) -> pd.DataFrame:
    """
    Demonstration pipeline assembling all textual innovation indicators.
    """

    df = metrics_df.copy()

    # Textual novelty
    df["novelty_raw"] = compute_textual_novelty(df)

    # Textual consolidation
    df["consolidation_raw"] = compute_textual_consolidation(df)

    # Textual disruption
    df["textual_disruption"] = compute_textual_disruption(
        df["novelty_raw"],
        df["consolidation_raw"]
    )

    # Build word frequency dictionary
    word_freq = word_freq_df["Word"].value_counts().to_dict()

    # Combinational novelty
    combo_df = compute_combinational_novelty_uzzi(
        word_freq=word_freq,
        path_word_combs=path_word_combs
    )

    # Merge all indicators
    df = df.merge(combo_df, on="PaperID", how="left")

    return df
