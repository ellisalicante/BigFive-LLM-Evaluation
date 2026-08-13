# Personality Without Persons?
**A Psychometric Critique of Big Five Testing in Large Language Models**


This repository contains the paper, appendix, code, data, and figures for the paper "Personality Without Persons?".

<br>

<p>
  <img src = "doc/figs/readme/line-1.png" height = "40">
</p>

<br>

## Paper aim

The paper evaluates whether Big Five questionnaires, originally designed for humans, can be applied to LLMs. It tests whether three defining characteristics of a personality trait are met.
It examines three RQs:
- **RQ1** Are human personality inventory (Big Five) items appropriate **descriptive summaries** of LLMs?
- **RQ2** Do personality scores capture meaningful **inter-model differences** across LLMs?
- **RQ3** Do LLMs' Big Five responses reflect **internal factors** consistent with the Big Five structure?

We assess the content validity of five candidate Big Five inventories and administer the best-performing inventory to $N = 264$ LLMs spanning 50 model families.

<br>

<p>
  <img src = "doc/figs/readme/line-1.png" height = "40">
</p>

<br>

## Repository structure

The repo is organized into the data (`dat`), the experiments (`exp`), the helper functions and source (`src`), and everything regarding the paper (document; `doc`).


<p>
  <img src = "doc/figs/readme/line-2.png" height = "40">
</p>


### Data ─ `dat`
contains the Big Five inventories (`00_inventories/`), expert rating data (`01_content_validity/`), prompt/instruction templates (`02_pilot_study/`), and final dataframes with all model information (`03_large_scale_administration/`):   

`final_dfs/` contains
- **`df_A` Item-Level Averages**  
  Models responses to individual items: mean scores and standard deviations across repetitions, contains item metadata.  
  *Format:* One row per item x model.
- **`df_B` Trait Scores**  
  Overall Big Five scores for all 247 models, averaged across all items and repetitions.  
  *Format:* One row per model, OCEAN dimensions as columns.
- **`df_metadata` Trait Scores + Metadata**  
  OCEAN scores from `df_B` with additional model metadata (e.g., parameter counts, release dates, reasoning support, and license type).
- **`df_cfa` Factor Analysis Matrix**  
  Models responses to individual items in wide-format, formatted for CFA in R.
  *Format:* Models as rows, item IDs as columns.
- **`df_lmm` Linear Mixed Model Data**  
  Model responses for all items and 5 repetitions and predictors for LMM (e.g., parameter size group, days since release, reasoning capability, license).  
  *Format:* One row per response (item x rep), indexed by model and item ID. Response ($y$) and predictors as columns.


<p>
  <img src = "doc/figs/readme/line-2.png" height = "40">
</p>


### Document ─ `doc`
contains the final paper pdf (`...`), the additional appendix (`...`), and all figures created for the paper (`figs/`).


<p>
  <img src = "doc/figs/readme/line-2.png" height = "40">
</p>


### Experiments ─ `exp`
contains all experiments that were run for this study. 
**RQ1** is tackled in `01_content_validity/`, and **RQ2** and **RQ3** in `03_large_scale_administration/`.

<br>

**1) Expert evaluation** `01_content_validity/`   
contains the script to evaluate content validity metrics of expert ratings for Big Five items.

- **RQ1** 
  - `eval_content_validity.py`

<br>

**2) Pilot study** `02_pilot_study/`   
contains the scripts to run and evaluate the pilot study, and the model responses.

- **Data collection**
  - `run_pilot_study.py` 
  - `responses_pilot/` 

- **Evaluation**
  - `eval_pilot_study.py` 

<br>

**3) Large-scale administration** `03_large_scale_administration/`   
contains the scripts for running and evaluating the large-scale administration of $N = 264$ models, spanning 50 model families, and the model responses.   

- **Data collection**
  - `run_api_models.py` API models  
  - `run_local_models.py` Huggingface models   
  - `responses/` individual csv files with data from all models

- **RQ2** 
  - `eval_norms.ipynb` Sample descriptives and norms
  - `eval_LMM.R` Linear mixed model and variance decomposition

- **RQ3**
  - `eval_CFA.R` confirmatory (CFA) and exploratory factor analysis (EFA)

- **Secondary analyses**
  - `eval_subgroups.ipynb` additional subgroup analyses and visualizations

<p>
  <img src = "doc/figs/readme/line-2.png" height = "40">
</p>


### Helpers and source code ─ `src`
contains the helper functions and source code for the experiments.   
This includes 
- functions for calculating the **content validity metrics** for the expert study (`content_validity.py`), 
- helpers for the **data collection** using APIs and huggingface (`API_prompting.py`, `huggingface_prompting.py`), 
- files for **reading** and **preprocessing** all models responses (`reading_data.py`, `preprocessing.py`), 
- and helpers for **visualizing** the results (`visualizations.py`).

<br>

<p>
  <img src = "doc/figs/readme/line-1.png" height = "40">
</p>

<br>

## Links

Our paper was accepted to _AIES 2026_.   
Read our paper on arXiv: https://arxiv.org/abs/2607.02325

