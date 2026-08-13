# Personality Without Persons?
**A Psychometric Critique of Big Five Testing in Large Language Models**


This repository contains the paper, appendix, code, data, and figures for the paper "Personality Without Persons?".

<br>

[//]: # (---)

<br>

## Paper aim

The paper evaluates whether Big Five questionnaires, originally designed for humans, can be applied to LLMs. It tests whether three defining characteristics of a personality trait are met.
It examines three RQs:
- **RQ1** Are human personality inventory (Big Five) items appropriate **descriptive summaries** of LLMs?
- **RQ2** Do personality scores capture meaningful **inter-model differences** across LLMs?
- **RQ3** Do LLMs' Big Five responses reflect **internal factors** consistent with the Big Five structure?

We assess the content validity of five candidate Big Five inventories and administer the best-performing inventory to $N = 264$ LLMs spanning 50 model families.

<br>

[//]: # (---)

<br>

## Repository structure

The repo is organized into the data (`dat`), the experiments (`exp`), the helper functions and source (`src`), and everything regarding the paper (document; `doc`).

<br>

[//]: # (---)

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


<br>

### Document ─ `doc`
contains the final paper pdf (`...`), the additional appendix (`...`), and all figures created for the paper (`figs/`).


[//]: # (---)

<br>

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


[//]: # (- `run_api_models.py`: Data collection: API models.)

[//]: # (- `run_local_models.py`: Data collection: local/open-weight models.)

[//]: # (- `eval_CFA.R`: Runs CFA and EFA.)

[//]: # (- `eval_LMM.R`: Runs linear mixed-effects models for variance decomposition and subgroup analyses.)

[//]: # (- `eval_norms.ipynb`: Descriptive analyses and norms.)

[//]: # (- `eval_subgroups.ipynb`: Subgroup comparisons.)

[//]: # (for **collecting the data** of .)

[//]: # (- `eval_norms.ipynb`   )

[//]: # (for **sample descriptives** and evaluating the **norms** of all models &#40;**RQ2**&#41;.)

[//]: # (- `eval_LMM.R`   )

[//]: # (for running a **linear mixed model** and computing the **variance decomposition** &#40;**RQ2**&#41;.)

[//]: # (- `eval_CFA.R`   )

[//]: # (for running a **confirmatory factor analysis** &#40;CFA&#41; and **exploratory factor analysis** &#40;EFA&#41; &#40;**RQ3**&#41;.)

[//]: # (- `eval_subgroups.ipynb`   )

[//]: # (for additional **subgroup analyses** and **visualizations** used in the paper.)


[//]: # (  * two `.py`-files for running API models and local models)

[//]: # (  * two `.ipynb`-files for evaluating the norms and group comparisons)

[//]: # (  * two `.R`-files for running the CFA and LMM analyses.)

[//]: # (---)

<br>

### Helpers and source code ─ `src`
contains the helper functions and source code for the experiments.   
This includes 
- functions for calculating the **content validity metrics** for the expert study (`content_validity.py`), 
- helpers for the **data collection** using APIs and huggingface (`API_prompting.py`, `huggingface_prompting.py`), 
- files for **reading** and **preprocessing** all models responses (`reading_data.py`, `preprocessing.py`), 
- and helpers for **visualizing** the results (`visualizations.py`).

<br>

[//]: # (---)
<p>
  <img src = "doc/figs/line.png" height = "50">
</p>

<br>

## Links

Our paper was accepted to _AIES 2026_.   
Read our paper on arXiv: https://arxiv.org/abs/2607.02325



[//]: # (```text)

[//]: # (dat/)

[//]: # (├── 00_inventories/)

[//]: # (│   ├── bfi-llm.json)

[//]: # (│   ├── lmlpa.json)

[//]: # (├── 01_content_validity/)

[//]: # (│   └── expert_ratings.csv)

[//]: # (├── 02_pilot_study/)

[//]: # (│   └── prompt-templates.json)

[//]: # (└── 03_large_scale_administration/)

[//]: # (    └── meta_info_models.csv)

[//]: # ()
[//]: # (exp/)

[//]: # (├── 01_content_validity/)

[//]: # (│   └── eval_content_validity.py)

[//]: # (├── 02_pilot_study/)

[//]: # (│   ├── responses_pilot/)

[//]: # (│   ├── eval_pilot_study.py)

[//]: # (│   └── run_pilot_study.py)

[//]: # (├── 03_large_scale_administration/)

[//]: # (    ├── responses/)

[//]: # (    ├── eval_CFA.R)

[//]: # (    ├── eval_LMM.R)

[//]: # (    ├── eval_norms.ipynb)

[//]: # (    ├── eval_subgroups.ipynb)

[//]: # (    ├── run_api_models.py)

[//]: # (    └── run_local_models.py)

[//]: # ()
[//]: # (src/)

[//]: # (├── API_prompting.py)

[//]: # (├── content_validity_metrics.py)

[//]: # (├── huggingface_prompting.py)

[//]: # (├── preprocessing.py)

[//]: # (├── reading_data.py)

[//]: # (└── visualizations.py)

[//]: # ()
[//]: # (doc/)

[//]: # (├── figs/)

[//]: # (└── tables/)

[//]: # (    └── descriptives_table.txt)

[//]: # ()
[//]: # (```)



[//]: # (## Data files)

[//]: # ()
[//]: # (### `dat/00_inventories/`)

[//]: # (Contains the candidate inventories used in the study.)

[//]: # ()
[//]: # (- `bfi-llm.json`: Winning Big Five inventory.)

[//]: # (- `lmlpa.json`: Alternative inventory used pilot.)

[//]: # ()
[//]: # (### `dat/01_content_validity/`)

[//]: # (- `expert_ratings.csv`: Expert ratings of items.)

[//]: # ()
[//]: # (### `dat/02_pilot_study/`)

[//]: # (- `prompt-templates.json`: Prompt formats tested in the pilot.)

[//]: # ()
[//]: # (### `dat/03_large_scale_administration/`)

[//]: # (- `meta_info_models.csv`: Metadata for the model sample.)

[//]: # ()
[//]: # (## Analysis code)

[//]: # ()
[//]: # (### `exp/01_content_validity/`)

[//]: # (- `eval_content_validity.py`: Evaluating content validity of items with expert ratings.)

[//]: # ()
[//]: # (### `exp/02_pilot_study/`)

[//]: # (- `run_pilot_study.py`: Data collection: Runs pilot using seven different prompt templates.)

[//]: # (- `eval_pilot_study.py`: Evaluates pilot outputs to identify best prompt template.)

[//]: # ()
[//]: # (### `exp/03_large_scale_administration/`)

[//]: # (- `run_api_models.py`: Data collection: API models.)

[//]: # (- `run_local_models.py`: Data collection: local/open-weight models.)

[//]: # (- `eval_CFA.R`: Runs CFA and EFA.)

[//]: # (- `eval_LMM.R`: Runs linear mixed-effects models for variance decomposition and subgroup analyses.)

[//]: # (- `eval_norms.ipynb`: Descriptive analyses and norms.)

[//]: # (- `eval_subgroups.ipynb`: Subgroup comparisons.)

[//]: # ()
[//]: # ()
[//]: # (## Helper functions)

[//]: # ()
[//]: # (### `src/`)

[//]: # ()
[//]: # ()
[//]: # (## Figures and tables)

[//]: # ()
[//]: # (### `doc/figs/`)

[//]: # (### `doc/tables/`)

[//]: # (- `descriptives_table.txt`: Per-model descriptive statistics.)
