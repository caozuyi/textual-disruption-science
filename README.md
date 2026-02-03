Replication code for figures and tables in the manuscript:

When disruptive science does not decline: Textual innovation and the decoupling of idea generation and scientific recognition, 1900–2021


This repository provides the full analysis code used to generate the main text and Extended Data figures and tables reported in the manuscript.  
The code is shared to ensure transparency, replicability, and clarity of the empirical procedures underlying all reported results.


---

## 1. Overview

The empirical analysis is based on large-scale publication-level data constructed from OpenAlex metadata and full-text-derived indicators, and SciSciNet lake.  
Due to the scale of the underlying dataset (tens of millions of publications), the original computations required a high-performance computing (HPC) environment.

To facilitate external review and replication, this repository provides stream-safe, memory-efficient, and fully transparent Python scripts that reproduce:

- all figures reported in the main text
- all figures and tables reported in the Extended Data section
- all regression specifications reported in the manuscript

No proprietary software is required.


---

## 2. Repository Structure

### /method/

Scripts for constructing the core analysis dataset and textual innovation indicators.

- build_meta_table.py  
  Builds the publication-level analysis table by merging bibliographic metadata, citation outcomes, and text-based indicators.

- build_textual_innovation_indicators.py  
  Computes raw and standardized measures of textual novelty, consolidation, and textual disruption from publication text.

These scripts define the core variables used throughout the analysis.


---

### /figures_and_tables/

Scripts used to generate all figures and tables reported in the paper.

Main text figures:

- figure2_ExtendedDataFig1_yearly_aggregation.py  
  Year-level aggregation of raw and standardized textual innovation indicators.  
  Used for Fig. 2a–b (main text) and Extended Data Fig. 1.

- figure3_period_regressions.py  
  Period-specific regression analyses examining shifts in the association between textual disruption and citation recognition across historical periods.

- figure4_ExtendedDataFig4_rolling_window.py  
  Rolling-window regression analyses estimating the time-varying association between textual disruption and citation recognition.  
  Used for Fig. 4 (main text), which reports long-run citation recognition based on ten-year citation counts (sci_C10), and Extended Data Fig. 4, which reports early citation recognition based on five-year citation counts (sci_C5).

Extended Data figures:

- ExtendedDataFig2_compute.py  
  Computes year-level trajectories of combinational novelty.

- ExtendedDataFig2_plot.py  
  Plots year-level and smoothed trajectories of combinational novelty.  
  Corresponds to Extended Data Fig. 2.

- ExtendedDataFig3_effect_size_trajectory.py  
  Estimates rolling-window effect-size trajectories, focusing on the magnitude and temporal stability of regression coefficients under alternative window-length specifications.  
  Corresponds to Extended Data Fig. 3.

Tables:

- Table1_EDT1_EDT2_STREAM_HC3.py  
  Stream-based regression script producing the results reported in Table 1, Extended Data Table 1, and Extended Data Table 2, using heteroskedasticity-robust (HC3) standard errors.


---

## 3. Data Availability

The underlying publication-level dataset is not included in this repository due to its scale and source-specific usage constraints associated with large bibliographic and full-text data.

All scripts assume the existence of a publication-level Parquet file (for example, meta_table.parquet) containing the variables documented in the manuscript.

The code is written such that:

- file paths can be adapted to local or cluster environments
- computation proceeds via chunked or streaming access
- no step requires loading the full dataset into memory


---

## 4. Computational Environment

- Python 3.8 or higher
- pandas
- numpy
- pyarrow
- statsmodels
- matplotlib
- scipy

The original analyses were executed in an HPC environment.  
The scripts provided here are platform-agnostic and can be executed on any system capable of handling large Parquet files via streaming.


---

## 5. Notes for Reviewers

- All figure- and table-level results reported in the manuscript can be traced directly to a corresponding script in this repository.
- The separation between data construction, analysis, and visualization is intentional and mirrors the structure of the empirical workflow.
- Early-period volatility in some indicators reflects sparse publication volumes rather than substantive innovation bursts, as discussed in the manuscript.


---

## 6. License

This code is released under the MIT License and may be used for academic and non-commercial research with appropriate citation.
