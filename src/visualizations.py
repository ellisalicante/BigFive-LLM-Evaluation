import os
import pandas as pd


import os
import pandas as pd

import os
import pandas as pd


def create_latex_descriptives_table(df_all, df_metadata, save_path="../../doc/tables/descriptives_table.txt"):
    """Create descriptive table and save as latex table."""

    import os
    import pandas as pd

    ### 1) DATA PREPARATION

    df_clean = df_all.dropna(subset=["score", "dimension"]).copy()

    df = df_all.copy()

    df["response_num"] = pd.to_numeric(df["response"], errors="coerce")
    df["is_valid"] = df["response_num"].isin([1, 2, 3, 4, 5])

    refusal_by_model = (
        df.groupby("model")
        .agg(
            total=("is_valid", "size"),
            valid=("is_valid", "sum")
        )
    )

    refusal_by_model["refusal_rate"] = 100 * (
        1 - (refusal_by_model["valid"] / refusal_by_model["total"])
    )

    refusal_by_model = refusal_by_model.reset_index()

    ### 2) MERGE METADATA

    meta = df_metadata[["model", "Model_name"]].drop_duplicates()

    df_clean = df_clean.merge(meta, on="model", how="left")
    refusal_by_model = refusal_by_model.merge(meta, on="model", how="left")

    ### 3) DESCRIPTIVES

    df_stats = (
        df_clean
        .groupby(["Model_name", "dimension"])["score"]
        .agg(["mean", "std"])
        .unstack("dimension")
    )

    df_stats.columns = [
        f"{stat}_{dim}" for stat, dim in df_stats.columns
    ]

    df_stats = df_stats.reset_index()

    df_final = df_stats.merge(
        refusal_by_model[["Model_name", "refusal_rate"]],
        on="Model_name",
        how="left"
    )

    df_final = df_final.sort_values("Model_name")

    ### 4) DIMENSIONS

    dims = [
        "Agreeableness",
        "Conscientiousness",
        "Extraversion",
        "Neuroticism",
        "Openness",
        "social-desirability"
    ]

    def fmt_mean(m):
        if pd.isna(m):
            return ""
        return f"{m:.2f}"

    def fmt_sd(s):
        if pd.isna(s):
            return ""
        return f"({s:.2f})"

    ### 5) ROWS

    rows = []

    for _, r in df_final.iterrows():

        mean_row = [
            r["Model_name"],
            f"{r['refusal_rate']:.1f}%" if pd.notna(r.get("refusal_rate")) else ""
        ]

        sd_row = ["", ""]

        for d in dims:
            mean_row.append(fmt_mean(r.get(f"mean_{d}")))
            sd_row.append(fmt_sd(r.get(f"std_{d}")))

        rows.append(" & ".join(mean_row) + " \\\\")
        rows.append(" & ".join(sd_row) + " \\\\")
        rows.append("\\hline")

    ### 6) TWO-COLUMN SAFE TABLE

    latex_table = r"""
\begin{table*}[t]
\centering
\begin{tabular}{lccccccc}
\hline
Model & Refusal (\%) & A & C & E & N & O & SD \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""

    ### 7) SAVE

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    return latex_table, df_clean, refusal_by_model