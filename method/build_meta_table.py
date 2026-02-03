#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data integration and harmonization pipeline
-------------------------------------------

This script constructs a unified paper-level meta-table by integrating
textual innovation measures derived from OpenAlex with citation-based
metrics from the SciSciNet database.

Key steps:
1. DOI normalization to ensure cross-database alignment
2. Mapping SciSciNet PaperIDs to DOIs
3. Linking citation-based disruption metrics (dc / dr) via DOI
4. Merging textual indicators with SciSciNet metadata
5. Exporting an analysis-ready meta-table

NOTE:
- This script documents the full data integration logic.
- Large-scale execution was performed on a high-performance computing (HPC)
  platform; this public version focuses on methodological transparency.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# =====================================================
# DOI normalization
# =====================================================

def normalize_doi(doi):
    """
    Standardize DOI strings to enable reliable cross-database matching.
    """
    if pd.isna(doi):
        return None
    x = str(doi).lower().strip()
    x = x.replace("https://doi.org/", "")
    x = x.replace("http://doi.org/", "")
    x = x.replace("doi:", "")
    return x if x else None


# =====================================================
# Paths (to be configured by user)
# =====================================================

OPENALEX_DIR = "PATH_TO_OPENALEX"
SCISCINET_DIR = "PATH_TO_SCISCINET"

papers_file  = f"{OPENALEX_DIR}/papers.csv"
textual_file = f"{OPENALEX_DIR}/textual_disruption_results.csv"

sci_file = f"{SCISCINET_DIR}/SciSciNet_Papers.tsv"
dc_file  = f"{SCISCINET_DIR}/pid_dc_summary.csv"
dr_file  = f"{SCISCINET_DIR}/pid_dr_summary.csv"


# =====================================================
# Step 1: Load SciSciNet metadata (DOI → full record)
# =====================================================

print("Loading SciSciNet metadata...")

sci_doi_map = {}
sci_cols = None

for chunk in pd.read_csv(sci_file, sep="\t", dtype=str, chunksize=200_000):
    chunk["DOI"] = chunk["DOI"].apply(normalize_doi)

    if sci_cols is None:
        sci_cols = list(chunk.columns)

    sub = chunk.dropna(subset=["DOI"])
    for _, row in sub.iterrows():
        sci_doi_map[row["DOI"]] = row.to_dict()

print("SciSciNet DOI records:", len(sci_doi_map))


# =====================================================
# Step 2: SciSciNet PaperID → DOI
# =====================================================

sci_pid_to_doi = {}

for chunk in pd.read_csv(sci_file, sep="\t", dtype=str, chunksize=200_000):
    doi = chunk["DOI"].apply(normalize_doi)
    for pid, d in zip(chunk["PaperID"], doi):
        if d:
            sci_pid_to_doi[pid] = d


# =====================================================
# Step 3: Load citation disruption (dc)
# =====================================================

print("Loading dc summary...")

dc_raw = pd.read_csv(dc_file, dtype=str)
dc_map = {}

for _, row in dc_raw.iterrows():
    pid = row["PaperID"]
    if pid in sci_pid_to_doi:
        dc_map[sci_pid_to_doi[pid]] = row["Value"]

print("dc mapped:", len(dc_map))


# =====================================================
# Step 4: Load citation disruption (dr)
# =====================================================

print("Loading dr summary...")

dr_raw = pd.read_csv(dr_file, dtype=str)
dr_map = {}

for _, row in dr_raw.iterrows():
    pid = row["PaperID"]
    if pid in sci_pid_to_doi:
        dr_map[sci_pid_to_doi[pid]] = row["Value"]

print("dr mapped:", len(dr_map))


# =====================================================
# Step 5: OpenAlex PaperID → DOI
# =====================================================

print("Loading OpenAlex paper identifiers...")

pid_to_doi = {}

for chunk in pd.read_csv(papers_file, dtype=str, chunksize=300_000):
    chunk["DOI"] = chunk["DOI"].apply(normalize_doi)
    for pid, doi in zip(chunk["PaperID"], chunk["DOI"]):
        if doi:
            pid_to_doi[pid] = doi

print("OpenAlex PID → DOI:", len(pid_to_doi))


# =====================================================
# Step 6: Build unified meta-table
# =====================================================

print("Building unified meta-table...")

rows = []

textual_cols = [
    "PaperID", "new_word", "new_word_reuse", "new_phrase", "new_phrase_reuse",
    "new_word_comb", "new_word_comb_reuse", "new_phrase_comb",
    "new_phrase_comb_reuse", "semantic_distance",
    "novelty_raw", "consolidation_raw",
    "Z_novelty", "Z_consolidation",
    "textual_disruption", "combo_novelty",
    "n_words", "n_phrases", "has_abstract"
]

for chunk in pd.read_csv(textual_file, dtype=str, chunksize=200_000):
    for _, row in chunk.iterrows():
        pid = row["PaperID"]
        if pid not in pid_to_doi:
            continue

        doi = pid_to_doi[pid]
        new_row = {}

        for col in textual_cols:
            new_row[col] = row.get(col, None)

        new_row["doi"] = doi
        new_row["openalex_pid"] = pid

        if doi in sci_doi_map:
            for col, val in sci_doi_map[doi].items():
                new_row[f"sci_{col}"] = val
        else:
            for col in sci_cols:
                new_row[f"sci_{col}"] = None

        new_row["dc"] = dc_map.get(doi, None)
        new_row["dr"] = dr_map.get(doi, None)

        rows.append(new_row)

print("Total rows in meta-table:", len(rows))


# =====================================================
# Step 7: Output
# =====================================================

df = pd.DataFrame(rows)

df.to_parquet("meta_table.parquet")
df.to_csv("meta_table.csv.gz", index=False)

print("Saved outputs:")
print("  meta_table.parquet")
print("  meta_table.csv.gz")
print("Shape:", df.shape)
