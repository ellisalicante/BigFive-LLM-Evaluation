# Personality Without Persons?
### A Psychometric Critique of Big Five Testing in Large Language Models


This repository contains the code, data, and figures for the AIES 2026 submission "Personality Without Persons?"


The project is organized around three stages of analysis:

1. **Content validity** of candidate personality inventories.
2. **Pilot study** to test prompts and response behavior.
3. **Large-scale administration** and statistical analysis across a broad model set.

## Repository structure

```text
dat/
├── 00_inventories/
│   ├── bfi-llm.json
│   ├── lmlpa.json
├── 01_content_validity/
│   └── expert_ratings.csv
├── 02_pilot_study/
│   └── prompt-templates.json
└── 03_large_scale_administration/
    └── meta_info_models.csv

exp/
├── 01_content_validity/
│   └── eval_content_validity.py
├── 02_pilot_study/
│   ├── responses_pilot/
│   ├── eval_pilot_study.py
│   └── run_pilot_study.py
├── 03_large_scale_administration/
    ├── responses/
    ├── eval_CFA.R
    ├── eval_LMM.R
    ├── eval_norms.ipynb
    ├── eval_subgroups.ipynb
    ├── run_api_models.py
    └── run_local_models.py

src/
├── API_prompting.py
├── content_validity_metrics.py
├── huggingface_prompting.py
├── preprocessing.py
├── reading_data.py
└── visualizations.py

doc/
├── figs/
└── tables/
    └── descriptives_table.txt

```

## Project aim

The paper evaluates whether standard Big Five questionnaires, originally designed for humans, can be applied to LLMs without losing psychometric meaning.
It examines three RQs:
- **RQ1** Are human personality inventory items (Big Five) appropriate **descriptive summaries** of LLMs?
- **RQ2** Do personality scores capture meaningful **inter-model differences** across LLMs?
- **RQ3** Do LLMs' Big Five responses reflect **internal factors** consistent with the Big Five structure?


## Data files

### `dat/00_inventories/`
Contains the candidate inventories used in the study.

- `bfi-llm.json`: Winning Big Five inventory.
- `lmlpa.json`: Alternative inventory used pilot.

### `dat/01_content_validity/`
- `expert_ratings.csv`: Expert ratings of items.

### `dat/02_pilot_study/`
- `prompt-templates.json`: Prompt formats tested in the pilot.

### `dat/03_large_scale_administration/`
- `meta_info_models.csv`: Metadata for the model sample.

## Analysis code

### `exp/01_content_validity/`
- `eval_content_validity.py`: Evaluating content validity of items with expert ratings.

### `exp/02_pilot_study/`
- `run_pilot_study.py`: Data collection: Runs pilot using seven different prompt templates.
- `eval_pilot_study.py`: Evaluates pilot outputs to identify best prompt template.

### `exp/03_large_scale_administration/`
- `run_api_models.py`: Data collection: API models.
- `run_local_models.py`: Data collection: local/open-weight models.
- `eval_CFA.R`: Runs CFA and EFA.
- `eval_LMM.R`: Runs linear mixed-effects models for variance decomposition and subgroup analyses.
- `eval_norms.ipynb`: Descriptive analyses and norms.
- `eval_subgroups.ipynb`: Subgroup comparisons.


## Helper functions

### `src/`


## Figures and tables

### `doc/figs/`
### `doc/tables/`
- `descriptives_table.txt`: Per-model descriptive statistics.
