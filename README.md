This repository provides the full analysis code used to generate the main and extended data figures and tables in the manuscript:



> \*When disruptive science does not decline: Textual innovation and the decoupling of idea generation and scientific recognition, 1900–2021\*



The code is shared to ensure transparency, replicability, and clarity of the empirical procedures underlying all reported results.



---



\## 1. Overview



The empirical analysis is based on large-scale publication-level data constructed from OpenAlex metadata and full-text-derived indicators.  

Due to the scale of the underlying dataset (tens of millions of publications), the original computations required a high-performance computing (HPC) environment.



To facilitate external review and replication, this repository provides \*\*stream-safe\*\*, \*\*memory-efficient\*\*, and \*\*fully transparent\*\* Python scripts that reproduce:



\- All reported figures in the main text

\- All figures and tables in the Extended Data section

\- All regression specifications reported in the manuscript



No proprietary software is required.



---



\## 2. Repository Structure



\### `/method/`



Scripts for constructing the core analysis dataset and textual innovation indicators.



\- \*\*`build_meta_table.py`\*\*  

&nbsp; Builds the publication-level analysis table by merging bibliographic metadata, citation outcomes, and text-based indicators.



\- \*\*`build_textual_innovation_indicators.py`\*\*  

&nbsp; Computes raw and standardized measures of textual novelty, consolidation, and textual disruption from publication text.



These scripts define the core variables used throughout the analysis.



---



\### `/figures\_and\_tables/`



Scripts used to generate all figures and tables reported in the paper.



\#### Main Text Figures



\- \*\*`figure2_ExtendedDataFig1_yearly_aggregation.py`\*\*  

&nbsp; Year-level aggregation of raw and standardized textual innovation indicators.  

&nbsp; Used for \*\*Fig. 2a–b (main text)\*\* and \*\*Extended Data Fig. 1\*\*.



\- \*\*`figure3_period_regressions.py`\*\*  

&nbsp; Period-specific regression analyses examining shifts in the relationship between textual disruption and citation impact.



\- \*\*`figure4_ExtendedDataFig4_rolling\_window.py`\*\*  

&nbsp; Rolling-window regressions estimating the time-varying effect of textual disruption on citation outcomes.  

&nbsp; Used for \*\*Fig. 4 (main text)\*\* and \*\*Extended Data Fig. 4\*\*.



---



\#### Extended Data Figures



\- \*\*`ExtendedDataFig2_compute.py`\*\*  

&nbsp; Computes year-level combination novelty trajectories.



\- \*\*`ExtendedDataFig2_plot.py`\*\*  

&nbsp; Plots the year-level and smoothed trajectories of combination novelty.  

&nbsp; Corresponds to \*\*Extended Data Fig. 2\*\*.



\- \*\*`ExtendedDataFig3_effect_size_trajectory.py`\*\*  

&nbsp; Estimates rolling-window effect-size trajectories under alternative window specifications.  

&nbsp; Corresponds to \*\*Extended Data Fig. 3\*\*.



---



\#### Tables



\- \*\*`Table1_EDT1_EDT2_STREAM_HC3.py`\*\*  

&nbsp; Stream-based regression script producing the main regression results reported in \*\*Table 1\*\*,  

&nbsp; \*\*Extended Data Table 1\*\*, and \*\*Extended Data Table 2\*\*, using heteroskedasticity-robust (HC3) standard errors.



---



\## 3. Data Availability



The underlying publication-level dataset is not included in this repository due to its scale and source-specific usage constraints associated with large bibliographic and full-text data.



All scripts assume the existence of a publication-level Parquet file (e.g., `meta\_table.parquet`) containing the variables documented in the manuscript.



The code is written such that:

\- File paths can be adapted to local or cluster environments

\- Computation proceeds via chunked or streaming access

\- No step requires loading the full dataset into memory



---



\## 4. Computational Environment



\- Python ≥ 3.8

\- pandas

\- numpy

\- pyarrow

\- statsmodels

\- matplotlib

\- scipy



The original analyses were executed on a university HPC cluster.  

However, the scripts provided here are platform-agnostic and can be executed on any system capable of handling large Parquet files via streaming.



---



\## 5. Notes for Reviewers



\- All figure- and table-level results reported in the manuscript can be traced directly to a corresponding script in this repository.

\- The separation between data construction, analysis, and visualization is intentional and mirrors the structure of the empirical workflow.

\- Early-period volatility in some indicators reflects sparse publication volumes rather than substantive innovation bursts, as discussed in the manuscript.



---



\## 6. License



This code is released under the MIT License and may be used for academic and non-commercial research with appropriate citation.



