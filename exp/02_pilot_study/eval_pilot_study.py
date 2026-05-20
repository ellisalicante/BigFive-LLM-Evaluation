


### 0) IMPORTS
from src.preprocessing import *



### 1) LOAD DATA
files = [
    "responses_pilot/gpt-4o.csv",
    "responses_pilot/claude-haiku-4-5.csv",
    "responses_pilot/deepseek-r1.csv",
    "responses_pilot/grok-4-1-fast-reasoning.csv",
    "responses_pilot/gemini-2-5-flash.csv",
    "responses_pilot/llama-3-1-8b-instruct.csv",
    "responses_pilot/glm-5.csv",
    "responses_pilot/qwen3-5-flash-02-23.csv",
    "responses_pilot/mistral-small-2603.csv",
]

# Concat all model dfs into one df
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


inventory_json = create_inventory_dict(
    "lmlpa.json",
    "bfi-llm.json",
    "social-desirability.json"
)

short_name_to_preamble, _ = get_short_name_dict()

# Add prompt template, dimension, key and recode
df = add_prompt_design(df, short_name_to_preamble)
df = add_dimension_key(df, inventory_json)
df = clean_NA_recode(df, 1, 5)

# Average over repetitions
df_avg = (
    df.groupby(
        ['model', 'inventory', 'item', 'prompt_design', 'situation']
    )
    .agg(
        response=('response', 'mean'),
        score=('score', 'mean'),
        score_std=('score', 'std'),
        dimension=('dimension', 'first'),
        key=('key', 'first'),
    )
    .reset_index()
)



### 2) RESULTS: PROMPT TEMPLATES
template_summary = pd.DataFrame({
    "Template": sorted(df["prompt_design"].unique())
})

template_summary["% Refusal"] = (
    template_summary["Template"]
    .map(refusal_rate(df, "prompt_design"))
)

template_summary["Variance"] = (
    template_summary["Template"]
    .map(variance(df_avg, "prompt_design"))
)

template_summary["Corr."] = (
    template_summary["Template"]
    .map(prompt_correlation(df_avg))
)

template_summary[r"$\Delta$ Keyed"] = (
    template_summary["Template"]
    .map(keyed_difference(df, "prompt_design"))
)

template_summary[r"$\alpha$"] = (
    template_summary["Template"]
    .map(cronbach_alpha(df_avg, "prompt_design"))
)



### 3) RESULTS: MODELS
model_summary = pd.DataFrame({
    "Model": sorted(df["model"].unique())
})

model_summary["% Refusal"] = (
    model_summary["Model"]
    .map(refusal_rate(df, "model"))
)


### 4) RESULTS: INVENTORY
df = df[df["inventory"].isin(["bfi-llm", "lmlpa"])].copy()

inventory_summary = pd.DataFrame({
    "Inventory": sorted(df["inventory"].unique())
})

inventory_summary["% Refusal"] = (
    inventory_summary["Inventory"]
    .map(refusal_rate(df, "inventory"))
)

inventory_summary["Variance"] = (
    inventory_summary["Inventory"]
    .map(variance(df_avg, "inventory"))
)

inventory_summary["Corr."] = (
    inventory_summary["Inventory"]
    .map(inventory_correlation(df_avg))
)

inventory_summary[r"$\alpha$"] = (
    inventory_summary["Inventory"]
    .map(cronbach_alpha(df_avg, "inventory"))
)



### 5) PRINT FINAL RESULTS
print("\n=== PROMPT TEMPLATE SUMMARY ===")
print(template_summary.to_string(index=False))

print("\n=== MODEL SUMMARY ===")
print(model_summary.to_string(index=False))

print("\n=== INVENTORY SUMMARY ===")
print(inventory_summary.to_string(index=False))