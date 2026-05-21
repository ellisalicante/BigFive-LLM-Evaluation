# =========================================================
# CREATE FINAL DATAFRAMES FOR ANALYSES
# =========================================================

import os
import re
import json
import numpy as np
import pandas as pd

from src.preprocessing import *


# =========================================================
# CONFIG
# =========================================================

response_dir = "responses"
save_dir = "../../dat/03_large_scale_administration/final_df"

target_prefix = "Here is a characteristic that may or may not apply to you"

os.makedirs(save_dir, exist_ok=True)

inventory_json = create_inventory_dict(
    "lmlpa.json",
    "bfi-llm.json",
    "social-desirability.json"
)


# =========================================================
# MODEL GROUPS
# =========================================================

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
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-0.6B",
}


# =========================================================
# CLEAN RESPONSE / REASONING
# =========================================================

def resolve_response_reasoning(resp_raw, reas_raw, model):

    resp_raw = "" if pd.isna(resp_raw) else str(resp_raw)
    reas_raw = "" if pd.isna(reas_raw) else str(reas_raw)

    def extract_1to5(text):
        m = re.search(r"\b([1-5])\b", text)
        return m.group(1) if m else ""

    def extract_json_score(text):
        m = re.search(r'"score"\s*:\s*"?([1-5])"?', text)
        return m.group(1) if m else ""

    if model in json_response_models:

        combined = resp_raw + "\n" + reas_raw

        response = extract_json_score(combined)

        if not response:
            response = extract_1to5(combined)

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


# =========================================================
# MAIN
# =========================================================

def create_final_dfs():

    dfs = []

    for fname in os.listdir(response_dir):

        if not fname.endswith(".csv"):
            continue

        df = pd.read_csv(
            os.path.join(response_dir, fname),
            low_memory=False
        )

        if "preamble" not in df.columns:
            continue

        df = df[
            df["preamble"]
            .astype(str)
            .str.startswith(target_prefix)
        ]

        if "rep" in df.columns:
            df = df.drop_duplicates(["prompt", "model", "rep"])
        else:
            df = df.drop_duplicates(["prompt", "model"])

        if len(df) < 580:
            continue

        df = process_df(df)

        df = add_dimension_key(df, inventory_json)
        df = clean_NA_recode(df, 1, 5)

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # -----------------------------------------------------
    # remove low-validity models
    # -----------------------------------------------------

    df_all["response"] = pd.to_numeric(
        df_all["response"],
        errors="coerce"
    )

    valid_counts = (
        df_all.groupby("model")["response"]
        .apply(lambda x: x.isin([1,2,3,4,5]).sum())
    )

    exclude_models = valid_counts[valid_counts < 200].index.tolist()

    df_all = df_all[
        ~df_all["model"].isin(exclude_models)
    ].copy()

    # -----------------------------------------------------
    # item ids
    # -----------------------------------------------------

    text_to_id = {
        item["text"]: item["id"]
        for item in inventory_json["items"]
    }

    # =====================================================
    # DF_A: item-level averaged scores
    # =====================================================

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

    df_A["item_id"] = df_A["item"].map(text_to_id)

    # =====================================================
    # DF_B: trait-level scores
    # =====================================================

    df_B = (
        df_all
        .groupby(["model", "dimension"])["score"]
        .mean()
        .unstack("dimension")
        .reset_index()
    )

    # =====================================================
    # DF_CFA: wide item matrix for CFA in R
    # =====================================================

    df_A["item_col"] = np.where(
        df_A["inventory"]
        .astype(str)
        .str.contains("social", case=False, na=False),
        "soc-" + df_A["item_id"].astype(int).astype(str).str.zfill(2),
        "bfi-" + df_A["item_id"].astype(int).astype(str).str.zfill(2)
    )

    # df_cfa = (
    #     df_A
    #     .pivot_table(
    #         index="model",
    #         columns="item_col",
    #         values="score",
    #         aggfunc="mean"
    #     )
    #     .sort_index(axis=1)
    # )

    df_bfi_only = df_A[df_A["item_col"].str.startswith("bfi-")].copy()

    df_cfa = (
        df_bfi_only
        .pivot_table(index="model", columns="item_col", values="score", aggfunc="mean")
        .sort_index(axis=1)
    )


    df_cfa_soc_des = (
        df_A
        .pivot_table(index="model", columns="item_col", values="score", aggfunc="mean")
        .sort_index(axis=1)
    )

    # =====================================================
    # DF_METADATA: trait scores + model metadata
    # =====================================================

    meta = pd.read_csv(
        "../../dat/03_large_scale_administration/meta_info_models.csv"
    ).rename(columns={"Model_ID": "model"})

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

    # =====================================================
    # DF_LMM: repeated-measures mixed-model dataframe (FIXED)
    # =====================================================

    df_lmm = df_all.copy()

    df_lmm["y"] = pd.to_numeric(df_lmm["score"], errors="coerce")

    # -----------------------------------------------------
    # SOCIAL DESIRABILITY (MODEL-LEVEL PREDICTOR)
    # -----------------------------------------------------

    df_soc = (
        df_lmm[df_lmm["dimension"] == "social-desirability"]
        .groupby("model", as_index=False)["y"]
        .mean()
        .rename(columns={"y": "y_soc_des"})
    )

    # -----------------------------------------------------
    # KEEP ONLY OCEAN TRAITS
    # -----------------------------------------------------

    OCEAN = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ]

    df_lmm = df_lmm[df_lmm["dimension"].isin(OCEAN)].copy()

    # -----------------------------------------------------
    # MERGE MODEL METADATA
    # -----------------------------------------------------

    # meta_model = (
    #     df_metadata[
    #         [
    #             "model",
    #             "Parameters_B",
    #             "Size",
    #             "Release_date",
    #             "Reasoning",
    #             "license_group"
    #         ]
    #     ]
    #     .drop_duplicates()
    # )

    def parse_params(val):
        try:
            return float(str(val).replace("B", "").strip())
        except:
            return np.nan

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

    # -----------------------------------------------------
    # MERGE SOCIAL DESIRABILITY (IMPORTANT FIX)
    # -----------------------------------------------------

    df_lmm = df_lmm.merge(df_soc, on="model", how="left")

    # -----------------------------------------------------
    # ITEM ID
    # -----------------------------------------------------

    text_to_id = {
        item["text"]: item["id"]
        for item in inventory_json["items"]
    }

    df_lmm["item_id"] = df_lmm["item"].map(text_to_id)

    # -----------------------------------------------------
    # REASONING / OPEN WEIGHT
    # -----------------------------------------------------

    df_lmm["Reasoning"] = (
        df_lmm["Reasoning"]
        .astype(str)
        .str.upper()
        .isin(["TRUE", "1", "YES"])
        .astype(int)
    )

    df_lmm["OpenWeight"] = (
        df_lmm["license_group"]
        .eq("open-weight")
        .astype(int)
    )

    # -----------------------------------------------------
    # RELEASE DATE
    # -----------------------------------------------------

    ref_date = pd.Timestamp("2026-05-13")

    df_lmm["ReleaseDate"] = (
            pd.to_datetime(df_lmm["Release_date"], errors="coerce")
            - ref_date
    ).dt.days.abs()

    # -----------------------------------------------------
    # SIZE
    # -----------------------------------------------------

    # df_lmm["Size"] = pd.to_numeric(df_lmm["Parameters_B"], errors="coerce")
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

    # -----------------------------------------------------
    # FINAL DF
    # -----------------------------------------------------

    df_lmm = (
        df_lmm[
            [
                "model",
                "item_id",
                "rep",
                "dimension",
                "y",
                "y_soc_des",
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

    # =====================================================
    # SAVE
    # =====================================================

    df_all.to_csv(f"{save_dir}/df_all.csv", index=False)
    df_A.to_csv(f"{save_dir}/df_A.csv", index=False)
    df_B.to_csv(f"{save_dir}/df_B.csv", index=False)
    df_cfa.to_csv(f"{save_dir}/df_cfa.csv")
    df_cfa_soc_des.to_csv(f"{save_dir}/df_cfa_soc_des.csv")
    df_metadata.to_csv(f"{save_dir}/df_metadata.csv", index=False)
    df_lmm.to_csv(f"{save_dir}/df_lmm.csv", index=False)

    # =====================================================
    # PRINT
    # =====================================================

    print("\nCreated dataframes:")
    print(f"df_all:      {df_all.shape}")
    print(f"df_A:        {df_A.shape}")
    print(f"df_B:        {df_B.shape}")
    print(f"df_cfa:      {df_cfa.shape}")
    print(f"df_cfa_soc_des: {df_cfa_soc_des.shape}")
    print(f"df_metadata: {df_metadata.shape}")
    print(f"df_lmm:      {df_lmm.shape}")

    return (
        df_all,
        df_A,
        df_B,
        df_cfa,
        df_cfa_soc_des,
        df_metadata,
        df_lmm
    )


# =========================================================
# RUN
# =========================================================

