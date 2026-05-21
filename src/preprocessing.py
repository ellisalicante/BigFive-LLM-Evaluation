import json
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
import pingouin as pg


def load_inventory(inventory_json: str):
    """Load inventory from .JSON file."""
    path = "../../dat/00_inventories/" + inventory_json

    with open(path) as f:
        inventory = json.load(f)

    return inventory


def create_inventory_dict(inv1, inv2, inv3):
    """Create dict from three inventories."""
    inv1_json = load_inventory(inv1)
    inv2_json = load_inventory(inv2)
    inv3_json = load_inventory(inv3)

    inventory_json = {"items": inv1_json["items"] + inv2_json["items"] + inv3_json["items"]}

    return inventory_json


def get_short_name_dict():
    """Get short name of prompt template mapped to item preamble and postamble."""
    item_preambles = []
    item_postambles = []
    item_short_names = []

    with open("../../dat/02_pilot_study/prompt-templates.json", "r") as f:
        instructions_json = json.load(f)

    for item_dict in instructions_json["items"]:
        item_preambles.append(item_dict["item_preamble"])
        item_postambles.append(item_dict["item_postamble"])
        item_short_names.append(item_dict["short_name"])

    # Create dictionary mapping short_name to preamble and postamble
    short_name_to_preamble = {item_dict["short_name"]: item_dict["item_preamble"] for item_dict in
                              instructions_json["items"]}
    short_name_to_postamble = {item_dict["short_name"]: item_dict["item_postamble"] for item_dict in
                               instructions_json["items"]}

    return short_name_to_preamble, short_name_to_postamble


def add_prompt_design(df, short_name_to_preamble):
    """Add prompt design to df."""
    short_names = []

    for index, row in df.iterrows():
        preamble = row["preamble"]
        short_name = None
        for key in short_name_to_preamble.keys():
            if short_name_to_preamble[key] == preamble:
                short_name = key
                break
        short_names.append(short_name)

    df["prompt_design"] = short_names

    return df


def add_dimension_key(df, inventory_json):
    """Add dimension & key to df."""
    item_lookup = {
        entry["text"]: {
            "dimension": entry["dimension"],
            "keying": entry["keying"]
        }
        for entry in inventory_json["items"]
    }

    df["dimension"] = df["item"].map(lambda x: item_lookup.get(x, {}).get("dimension"))
    df["key"] = df["item"].map(lambda x: item_lookup.get(x, {}).get("keying"))

    return df


def get_df_out_of_range(df, scale_min, scale_max):
    """Get out of range values."""
    non_numeric_mask = pd.to_numeric(df["response"], errors="coerce").isna()
    non_numeric = df[non_numeric_mask]
    numeric = df[~non_numeric_mask]

    df_non_num = df[~non_numeric_mask].copy()
    df_non_num["response"] = df_non_num["response"].astype(float)

    out_of_range_mask = ~df_non_num["response"].between(scale_min, scale_max)
    out_of_range = df_non_num[out_of_range_mask]
    in_range = df_non_num[~out_of_range_mask]

    print(len(df))
    print("Number of responses that are non-numeric:", len(non_numeric), f"({len(non_numeric)/len(df)*100:.2f}%)")
    print(f"Number of responses that are out of range (min {scale_min}, max {scale_max}):", len(out_of_range), f"({len(out_of_range)/len(df)*100:.2f}%)")

    return numeric, non_numeric, in_range, out_of_range


def clean_NA_recode(df, scale_min, scale_max):
    """Replace all responses that are not numeric or not between scale_min and scale_max with NA"""
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df["response"] = df["response"].where(df["response"].between(scale_min, scale_max))

    df["score"] = np.where(
        df["key"] == "-",
        scale_min + scale_max - df["response"],
        df["response"]
    )

    return df


def calculate_score_recode(df_numeric, df_in_range, scale_min, scale_max):
    """Calculate the score based on whether responses need to be recoded (for reverse-keyed items)."""
    df_numeric["response"] = df_numeric["response"].astype(float)
    df_in_range["response"] = df_in_range["response"].astype(float)

    df_numeric["score"] = np.where(
        df_numeric["key"] == "-",
        scale_min + scale_max - df_numeric["response"],
        df_numeric["response"]
    )
    df_in_range["score"] = np.where(
        df_in_range["key"] == "-",
        scale_min + scale_max - df_in_range["response"],
        df_in_range["response"]
    )

    return df_numeric, df_in_range



def refusal_rate(df, group):
    """Calculate the refusal rate."""
    total = df[group].value_counts()
    missing = df[df["response"].isna()][group].value_counts()
    return round((missing / total * 100).fillna(0), 2)


def variance(df_avg, group):
    """Calculate the variance."""
    return round(df_avg.groupby(group)["score_std"].mean(), 2)


def keyed_difference(df, group):
    """Calculate the difference between positively and negatively keyed items."""
    dim_key = (
        df.groupby(
            ['model', group, 'inventory', 'dimension', 'key']
        )['score']
        .mean()
        .reset_index()
    )

    pivot = dim_key.pivot_table(
        index=['model', group, 'inventory', 'dimension'],
        columns='key',
        values='score'
    ).reset_index()

    summary = pivot.groupby(group)[['+', '-']].mean()

    return round((summary["+"] - summary["-"]).abs(), 2)


def inventory_correlation(df_avg):
    """Calculate the mean correlation between all possible combinations of prompt templates."""
    inv_means = (
        df_avg.groupby(
            ['model', 'prompt_design', 'inventory'],
            as_index=False
        )['score']
        .mean()
    )

    inv_means = inv_means[inv_means["inventory"].isin(["bfi-llm", "lmlpa"])].copy()

    rows = []

    for inventory, grp in inv_means.groupby('inventory'):

        wide = grp.pivot_table(
            index='model',
            columns='prompt_design',
            values='score'
        )

        rhos = []

        for p1, p2 in combinations(wide.columns, 2):

            col = wide[[p1, p2]].dropna()

            if len(col) < 3:
                continue

            rho, _ = spearmanr(col[p1], col[p2])

            if not np.isnan(rho):
                rhos.append(rho)

        rows.append({
            "inventory": inventory,
            "Corr.": round(np.mean(rhos), 2)
        })

    return pd.DataFrame(rows).set_index("inventory")["Corr."]


def prompt_correlation(df_avg):
    """Calculate the mean correlation between BFI-LLM and LMLPA."""
    df_agg = (
        df_avg.groupby(
            ['model', 'inventory', 'dimension', 'prompt_design']
        )['score']
        .mean()
        .reset_index()
    )

    rows = []

    for prompt in df_agg["prompt_design"].unique():

        cors = []

        for dim in [
            "Openness",
            "Conscientiousness",
            "Extraversion",
            "Agreeableness",
            "Neuroticism"
        ]:

            sub = df_agg[
                (df_agg["prompt_design"] == prompt) &
                (df_agg["dimension"] == dim)
            ]

            pivot = (
                sub.pivot(
                    index='model',
                    columns='inventory',
                    values='score'
                )
                .dropna()
            )

            if (
                "bfi-llm" in pivot.columns and
                "lmlpa" in pivot.columns and
                len(pivot) > 1
            ):

                rho, _ = spearmanr(
                    pivot["bfi-llm"],
                    pivot["lmlpa"]
                )

                cors.append(rho)

        rows.append({
            "prompt_design": prompt,
            "Corr.": round(np.mean(cors), 2)
        })

    return pd.DataFrame(rows).set_index("prompt_design")["Corr."]


def cronbach_alpha(df_avg, group):
    """Calculate Cronbach's alpha."""
    rows = []

    for name, grp in df_avg.groupby(group):

        wide = grp.pivot_table(
            index='model',
            columns='item',
            values='score'
        )

        try:
            alpha, _ = pg.cronbach_alpha(wide)
            alpha = round(alpha, 2)

        except:
            alpha = np.nan

        rows.append({
            group: name,
            r'$\alpha$': alpha
        })

    return pd.DataFrame(rows).set_index(group)[r'$\alpha$']

