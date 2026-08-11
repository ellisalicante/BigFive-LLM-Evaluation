### 0) IMPORTS
import os
import re
from src.preprocessing import *


### 1) CONFIG
response_dir = "responses"

save_dir = "../../dat/03_large_scale_administration/final_dfs"
os.makedirs(save_dir, exist_ok=True)

target_prefix = "Here is a characteristic that may or may not apply to you"

meta = pd.read_csv(
    "../../dat/03_large_scale_administration/meta_info_models.csv"
).rename(columns={"Model_ID": "model"})

inventory_json = create_inventory_dict(
    "lmlpa.json",
    "bfi-llm.json",
    "social-desirability.json"
)

OCEAN = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism"
]


# 2) MODEL GROUPS
# -> patterns in how different models responded
think_response_models = {
    "deepinfra/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepinfra/deepseek-ai/DeepSeek-R1-0528-Turbo",
    "deepinfra/deepseek-ai/DeepSeek-R1-0528",
    "deepinfra/Qwen/Qwen3-30B-A3B",
    "deepinfra/Qwen/Qwen3-14B",
    "deepinfra/Qwen/Qwen3-32B",
    "deepinfra/google/gemini-2.5-flash",
    "deepinfra/google/gemini-2.5-pro",
}

response_other = {
    "deepinfra/microsoft/phi-4",
    "openrouter/mancer/weaver",
}

response_explanation_models = {
    "openrouter/mistralai/mistral-7b-instruct-v0.1",
    "xai/grok-4.20",
    "openrouter/ibm-granite/granite-4.1-8b"
}

response_in_reas_models = {
    "deepinfra/Qwen/Qwen3.5-0.8B",
    "deepinfra/Qwen/Qwen3.5-2B"
}

reasoning_with_response = {
    "openrouter/minimax/minimax-m1"
}

json_response_models = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B-Base",
    "google/gemma-4-E2B",
    "google/gemma-4-E2B-it",
    "google/gemma-3-4b-pt",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2-7b",
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b", #!
    "google/gemma-7b-it", #!
    "Qwen/Qwen2.5-7B", #!
    "Qwen/Qwen2.5-7B-Instruct", #!
    "google/gemma-3-1b-it",
    "google/gemma-3-1b-pt",
    "google/gemma-3-270m",
    "google/gemma-3-270m-it",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "tiiuae/Falcon3-1B-Base",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-3B-Base",
    "tiiuae/Falcon3-3B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "ibm-granite/granite-4.0-h-micro-base",
    "allenai/Olmo-Hybrid-7B", #!
    "allenai/Olmo-Hybrid-Instruct-SFT-7B" #!
    "ThingAI/Quark-270m-Instruct",
    "ThingAI/Quark-270m-Base",
    "deepseek-ai/DeepSeek-V4-Flash-Base",
    "deepseek-ai/DeepSeek-V4-Pro-Base",
    "XiaomiMiMo/MiMo-V2.5-Pro-Base",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen/Qwen3-14B-Base",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
    "mistralai/Mistral-Nemo-Base-2407",
    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mixtral-8x22B-v0.1",
    "allenai/OLMo-2-0425-1B",
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-0325-32B",
    "allenai/OLMo-2-0325-32B-Instruct",
    "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "arcee-ai/Trinity-Large-TrueBase",
    "arcee-ai/Trinity-Mini-Base-Pre-Anneal",
    "EssentialAI/rnj-1",
    "mistralai/Ministral-3-14B-Base-2512",
    "mistralai/Ministral-3-8B-Base-2512",
    "mistralai/Ministral-3-3B-Base-2512",
    "tencent/Hy3-preview-Base",
    "tencent/Hunyuan-A13B-Pretrain",
    "baidu/ERNIE-4.5-300B-A47B-Base-PT",
    "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT",
    "google/gemma-4-31B",
    "google/gemma-2-27b",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.2-11B-Vision",
    "meta-llama/Llama-3.2-90B-Vision",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
}


# 3) HELPERS
# For cleaning responses and reasoning

def extract_json_objects(text):
    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i+1])
                    start = None
    return objs


def parse_valid_score_jsons(text, strict_schema=True):
    valid = []
    for candidate in extract_json_objects(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "score" not in obj or "response" not in obj:
            continue
        if strict_schema and set(obj.keys()) != {"score", "response"}:
            continue
        try:
            score_int = int(obj["score"])
        except (TypeError, ValueError):
            continue
        if not (1 <= score_int <= 5):
            continue
        valid.append(obj)
    return valid


def resolve_response_reasoning(resp_raw, reas_raw, model):
    resp_raw = "" if pd.isna(resp_raw) else str(resp_raw)
    reas_raw = "" if pd.isna(reas_raw) else str(reas_raw)

    def extract_1to5(text):
        m = re.search(r"\b([1-5])\b", text)
        return m.group(1) if m else ""

    if model in json_response_models:
        combined = resp_raw + "\n" + reas_raw
        valid_jsons = parse_valid_score_jsons(combined)
        if len(valid_jsons) == 1:
            response = str(valid_jsons[0]["score"])
        else:
            # zero valid JSONs (refusal/garbage) OR more than one JSON object
            # (model didn't follow "exactly one JSON" instruction) -> treat as NA
            response = "NA"
        return response, combined.strip()

    if model in response_explanation_models or model in response_other:
        return extract_1to5(resp_raw + " " + reas_raw), ""

    if model in think_response_models:
        response = extract_1to5(resp_raw)
        if not response:
            response = extract_1to5(reas_raw)
        reasoning = reas_raw.strip() if reas_raw else resp_raw.strip()
        return response, reasoning

    if model in response_in_reas_models:
        return extract_1to5(resp_raw + " " + reas_raw), resp_raw or reas_raw

    if model in reasoning_with_response:
        response = extract_1to5(reas_raw)
        lines = [l.strip() for l in reas_raw.splitlines() if l.strip()]
        if lines and re.search(r"\b([1-5])\b", lines[-1]):
            reasoning = "\n".join(lines[:-1]).strip()
        else:
            reasoning = reas_raw.strip()
        return response, reasoning

    return extract_1to5(resp_raw + " " + reas_raw), reas_raw or ""


def process_df(df):

    parsed = df.apply(
        lambda row: resolve_response_reasoning(
            row.get("response", ""),
            row.get("reasoning", ""),
            row["model"]
        ),
        axis=1
    )

    df["response"], df["reasoning"] = zip(*parsed)

    return df


def parse_params(val):
    try:
        return float(str(val).replace("B", "").strip())
    except:
        return np.nan


# 4) CREATE FINAL DFs
def create_final_dfs():

    # Read all response files and keep only valid, complete runs
    dfs = []
    for fname in os.listdir(response_dir):

        if not fname.endswith(".csv"):
            continue

        df = pd.read_csv(
            os.path.join(response_dir, fname),
            low_memory=False
        )

        df = df[
            df["preamble"]
            .astype(str)
            .str.startswith(target_prefix)
        ]

        if "rep" in df.columns:
            df = df.drop_duplicates(["prompt", "model", "rep"])
        else:
            df = df.drop_duplicates(["prompt", "model"])

        # Sanity check
        if len(df) < 580:
            continue

        df = process_df(df)
        df = add_dimension_key(df, inventory_json)
        df = clean_NA_recode(df, 1, 5)

        dfs.append(df)

    df_all_raw = pd.concat(dfs, ignore_index=True)


    # Remove models with less than 200 valid responses
    df_all_raw["response"] = pd.to_numeric(
        df_all_raw["response"],
        errors="coerce"
    )

    all_models_before = set(df_all_raw["model"].unique())

    valid_counts = (
        df_all_raw.groupby("model")["response"]
        .apply(lambda x: x.isin([1, 2, 3, 4, 5]).sum())
    )

    exclude_models = valid_counts[valid_counts < 200].index.tolist()

    print(f"\nModels excluded (< 200 valid responses), {len(exclude_models)} total:")
    excluded_summary = (
        valid_counts[valid_counts < 200]
        .sort_values()
        .rename("n_valid")
        .to_frame()
    )
    print(excluded_summary.to_string())

    df_all_raw = df_all_raw[df_all_raw["dimension"].isin(OCEAN)]

    df_all = df_all_raw[
        ~df_all_raw["model"].isin(exclude_models)
    ].copy()

    print(f"\nModels present before exclusion: {len(all_models_before)}")
    print(f"Models remaining after exclusion: {df_all['model'].nunique()}")


    # DF_A
    # Item-level averaged scores
    df_A = (
        df_all
        .groupby(['model', 'inventory', 'item', 'situation'])
        .agg(
            prompt=('prompt', 'first'),
            preamble=('preamble', 'first'),
            postamble=('postamble', 'first'),
            options=('options', 'first'),
            timestamp=('timestamp', 'first'),
            usage=('usage', 'first'),
            reasoning=('reasoning', 'first'),
            dimension=('dimension', 'first'),
            key=('key', 'first'),
            response=('response', 'mean'),
            score=('score', 'mean'),
            score_std=('score', 'std'),
        )
        .reset_index()
    )

    # Map item text to item id
    text_to_id = {
        item["text"]: item["id"]
        for item in inventory_json["items"]
    }

    df_A["item_id"] = df_A["item"].map(text_to_id)

    df_A["item_col"] = np.where(
        df_A["inventory"]
        .astype(str)
        .str.contains("social", case=False, na=False),
        "soc-" + df_A["item_id"].astype(int).astype(str).str.zfill(2),
        "bfi-" + df_A["item_id"].astype(int).astype(str).str.zfill(2)
    )


    # DF_B
    # Trait-level scores
    df_B = (
        df_all
        .groupby(["model", "dimension"])["score"]
        .mean()
        .unstack("dimension")
        .reset_index()
    )

    df_B = df_B[["model"] + OCEAN]


    # DF_CFA
    # Wide item matrix for CFA in R
    df_cfa = (
        df_A
        .pivot_table(index="model", columns="item_col", values="score", aggfunc="mean")
        .sort_index(axis=1)
    )


    # DF_METADATA
    # Trait scores + model metadata
    df_metadata = df_B.merge(
        meta,
        on="model",
        how="left"
    )

    df_metadata["license_group"] = df_metadata["License"].apply(
        lambda x:
        "open-weight"
        if isinstance(x, str)
        and "proprietary" not in x.lower()
        else "proprietary"
    )


    # DF_LMM
    # Dataframe for linear mixed model
    df_lmm = df_all.copy()
    df_lmm["y"] = pd.to_numeric(df_lmm["score"], errors="coerce")
    df_lmm = df_lmm[df_lmm["dimension"].isin(OCEAN)].copy()

    # Merge model metadata
    df_metadata["params_numeric"] = df_metadata["Parameters_B"].apply(parse_params)
    meta_model = df_metadata[
        [
            "model",
            "params_numeric",
            "Release_date",
            "Reasoning",
            "license_group"
        ]
    ].drop_duplicates()
    df_lmm = df_lmm.merge(meta_model, on="model", how="left")

    # Item id
    text_to_id = {
        item["text"]: item["id"]
        for item in inventory_json["items"]
    }
    df_lmm["item_id"] = df_lmm["item"].map(text_to_id)

    # Reasoning
    df_lmm["Reasoning"] = (
        df_lmm["Reasoning"]
        .astype(str)
        .str.upper()
        .isin(["TRUE", "1", "YES"])
        .astype(int)
    )

    # Open-weight
    df_lmm["OpenWeight"] = (
        df_lmm["license_group"]
        .eq("open-weight")
        .astype(int)
    )

    # Release date
    ref_date = pd.Timestamp("2026-05-13")
    df_lmm["ReleaseDate"] = (
            pd.to_datetime(df_lmm["Release_date"], errors="coerce")
            - ref_date
    ).dt.days.abs()

    # Size
    df_lmm["Size"] = pd.to_numeric(df_lmm["params_numeric"], errors="coerce")

    df_lmm["SizeGroup"] = pd.cut(
        df_lmm["Size"],
        bins=[-np.inf, 10, 100, np.inf],
        labels=["small", "medium", "large"]
    )

    df_lmm["SizeGroup"] = (
        df_lmm["SizeGroup"]
        .cat.add_categories("undisclosed")
        .fillna("undisclosed")
    )

    # Final df_lmm
    df_lmm = (
        df_lmm[
            [
                "model",
                "item_id",
                "rep",
                "dimension",
                "y",
                "Size",
                "SizeGroup",
                "ReleaseDate",
                "Reasoning",
                "OpenWeight"
            ]
        ]
        .rename(columns={"model": "model_id"})
        .dropna(subset=["y"])
    )

    df_lmm["model_id"] = df_lmm["model_id"].astype("category")


    # Save
    df_all_raw.to_csv(f"{save_dir}/df_all_raw.csv", index=False)
    df_all.to_csv(f"{save_dir}/df_all.csv", index=False)
    df_A.to_csv(f"{save_dir}/df_A.csv", index=False)
    df_B.to_csv(f"{save_dir}/df_B.csv", index=False)
    df_cfa.to_csv(f"{save_dir}/df_cfa.csv")
    df_metadata.to_csv(f"{save_dir}/df_metadata.csv", index=False)
    df_lmm.to_csv(f"{save_dir}/df_lmm.csv", index=False)

    # Print
    print("\nCreated dataframes:")
    print(f"df_all_raw:      {df_all_raw.shape}")
    print(f"df_all:      {df_all.shape}")
    print(f"df_A:        {df_A.shape}")
    print(f"df_B:        {df_B.shape}")
    print(f"df_cfa:      {df_cfa.shape}")
    print(f"df_metadata: {df_metadata.shape}")
    print(f"df_lmm:      {df_lmm.shape}")

    return (
        df_all_raw,
        df_all,
        df_A,
        df_B,
        df_cfa,
        df_metadata,
        df_lmm,
    )