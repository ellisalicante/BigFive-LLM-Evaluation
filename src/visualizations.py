### IMPORTS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import (
    ttest_rel, ttest_1samp, ttest_ind,
    mannwhitneyu, kruskal, kstest, zscore, probplot,
    levene, shapiro, linregress
)
from statsmodels.stats.multitest import multipletests
import itertools


### PALETTES & STYLE

IMPACT_SIX = ["#008CBB", "#9569D1", "#A3D900", "#E30053", "#FFB000", "#3092FF"]
IMPACT_TEN = [
    "#008CBB", "#9569D1", "#A3D900", "#E30053", "#FFB000",
    "#3092FF", "#00BFA5", "#FF6F61", "#6C5CE7", "#7CB342",
]
BINARY_PAL_BLUE_PURPLE = ["#008CBB", "#9569D1"]
BINARY_PAL_GREEN_RED   = ["#A3D900", "#E30053"]
BINARY_PAL_BLUE_ORANGE = ["#3092FF", "#FFB000"]
BINARY_PAL_BLUE_PINK = ["#008CBB", "#E30053"]
TRI_COLOR              = ["#E30053", "#FFB000", "#3092FF"]

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "impact_diverging",
    [IMPACT_SIX[3], "#FFFFFF", IMPACT_SIX[4]],
    N=256,
)

OCEAN_COLS = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness",
]

HUMAN_BFI_NORMS = {
    "Conscientiousness": 3.74,
    "Agreeableness":     3.95,
    "Neuroticism":       2.85,
    "Openness":          3.24,
    "Extraversion":      3.53,
}

FIG_WIDTH  = 7.0
FIG_HEIGHT = 3.5

FAMILY_LABELS = {
    "qwen": "Qwen",
    "claude": "Claude",
    "seed": "Seed",
    "other": "Other",
    "minimax": "MiniMax",
    "mimo": "MiMo",
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
    "gemma": "Gemma",
    "llama": "Llama",
    "mistral": "Mistral",
    "kimi": "Kimi",
    "nemotron": "Nemotron",
    "gpt-oss": "GPT-OSS",
    "glm": "GLM",
    "gpt": "GPT",
    "o-series": "O-Series",
    "aion": "Aion",
    "trinity": "Trinity",
    "ernie": "ERNIE",
    "sonar": "Sonar",
    "grok": "Grok",
}

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


def apply_paper_style():
    """Apply consistent matplotlib style for paper figures."""
    plt.rcParams.update({
        "figure.figsize":   (FIG_WIDTH, FIG_HEIGHT),
        "font.size":        22,
        "axes.titlesize":   22,
        "axes.labelsize":   22,
        "xtick.labelsize":  22,
        "ytick.labelsize":  22,
        "legend.fontsize":  22,
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlepad":    10,
        "axes.labelpad":    10,
    })


### STATISTICAL HELPERS

def descriptive_stats(df, cols=OCEAN_COLS):
    """Return mean, SD, min, Q1, median, Q3, max for each column."""

    desc = df[cols].agg(["mean", "std", "min", "max"]).T

    desc["Q1"] = df[cols].quantile(0.25)
    desc["Median"] = df[cols].quantile(0.50)
    desc["Q3"] = df[cols].quantile(0.75)

    return desc.rename(columns={
        "mean": "Mean",
        "std": "SD",
        "min": "Min",
        "max": "Max"
    })[["Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"]]


def ks_normality_tests(df, cols=OCEAN_COLS):
    """
    Kolmogorov-Smirnov normality test (z-scored) for each column.
    Returns a DataFrame with D, p, and interpretation.
    """
    rows = []
    for col in cols:
        data = df[col].dropna()
        z = zscore(data)
        ks_stat, ks_p = kstest(z, "norm")
        rows.append({
            "trait":          col,
            "D":              round(ks_stat, 3),
            "p":              round(ks_p, 3),
            "interpretation": "non-normal" if ks_p < 0.05 else "approx. normal",
        })
    return pd.DataFrame(rows)


def bonferroni_ttest(df, group_col, cols=OCEAN_COLS):
    """
    Independent-samples Welch t-test with Bonferroni correction
    for exactly two groups. Returns trait-level results.
    """
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("bonferroni_ttest requires exactly two groups.")
    g1, g2 = groups
    rows = []
    for col in cols:
        a = df[df[group_col] == g1][col].dropna()
        b = df[df[group_col] == g2][col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = ttest_ind(a, b, equal_var=False)
        p_b = min(p * len(cols), 1.0)
        rows.append({
            "trait":         col,
            "group1":        g1,
            "group2":        g2,
            "t":             round(t, 3),
            "p_bonferroni":  round(p_b, 4),
            "sig":           "***" if p_b < .001 else "**" if p_b < .01 else "*" if p_b < .05 else "ns",
        })
    return pd.DataFrame(rows)


def paired_ttest_base_vs_it(df_models, pairs, cols=OCEAN_COLS):
    """
    One-sample t-test on paired differences (instruction-tuned minus base).
    `pairs` is a list of (base_model_id, instruct_model_id) tuples.
    Returns per-trait mean_diff, SD, t, p.
    """
    results = []
    for base, inst in pairs:
        base_row = df_models[df_models["model"] == base]
        inst_row = df_models[df_models["model"] == inst]
        if base_row.empty or inst_row.empty:
            print(f"Missing pair: {base} vs {inst}")
            continue
        base_row = base_row.iloc[0]
        inst_row = inst_row.iloc[0]
        for trait in cols:
            results.append({
                "pair":        f"{base} → {inst}",
                "trait":       trait,
                "base":        base_row[trait],
                "instruction": inst_row[trait],
                "diff":        inst_row[trait] - base_row[trait],
            })
    df_pairs = pd.DataFrame(results)

    stats_rows = []
    for trait in cols:
        d = df_pairs[df_pairs["trait"] == trait]["diff"].dropna()
        t, p = ttest_1samp(d, 0)
        stats_rows.append({
            "trait":     trait,
            "mean_diff": round(d.mean(), 3),
            "sd_diff":   round(d.std(), 3),
            "t":         round(t, 3),
            "p":         round(p, 5),
        })
    print("=== Per-pair OCEAN differences (IT − Base) ===\n")

    pair_table = (
        df_pairs
        .pivot(index="pair", columns="trait", values="diff")
        .round(3)
    )

    # enforce OCEAN order
    pair_table = pair_table[
        ["Openness",
         "Conscientiousness",
         "Extraversion",
         "Agreeableness",
         "Neuroticism"]
    ]

    # short labels
    pair_table.columns = ["O", "C", "E", "A", "N"]

    print(pair_table.to_string())

    return df_pairs, pd.DataFrame(stats_rows)


def regression_summary(df, x_col, y_cols=OCEAN_COLS):
    """OLS regression of each trait on a continuous predictor."""
    rows = []
    for col in y_cols:
        tmp = df[[x_col, col]].dropna()
        slope, intercept, r, p, se = linregress(tmp[x_col], tmp[col])
        rows.append({
            "trait":   col,
            "slope":   round(slope, 5),
            "r":       round(r, 3),
            "p_value": round(p, 5),
            "sig":     "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns",
        })
    return pd.DataFrame(rows).sort_values("p_value")


def mannwhitney_pairwise(df, group_col, cols=OCEAN_COLS, groups=None):
    """
    Kruskal-Wallis + pairwise Mann-Whitney U (FDR-BH) for ≥ 2 groups.
    Returns global-test DataFrame and pairwise-comparison DataFrame.
    """
    if groups is None:
        groups = sorted(df[group_col].dropna().unique())

    global_rows = []
    pairwise_rows = []

    for col in cols:
        samples = [df[df[group_col] == g][col].dropna() for g in groups]

        if len(groups) == 2:
            stat, p = mannwhitneyu(samples[0], samples[1], alternative="two-sided")
            stat_name = "U"
        else:
            stat, p = kruskal(*samples)
            stat_name = "H"

        global_rows.append({
            "trait":     col,
            "stat_name": stat_name,
            "stat":      round(stat, 3),
            "p":         round(p, 5),
            "sig":       "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns",
        })

        if len(groups) > 2 and p < .05:
            pair_results = []
            for g1, g2 in itertools.combinations(groups, 2):
                x = df[df[group_col] == g1][col].dropna()
                y = df[df[group_col] == g2][col].dropna()
                u_pair, p_pair = mannwhitneyu(x, y, alternative="two-sided")
                pair_results.append({
                    "trait": col, "g1": g1, "g2": g2,
                    "U": u_pair, "p_raw": p_pair,
                    "m1": x.mean(), "m2": y.mean(),
                })
            corrected = multipletests([r["p_raw"] for r in pair_results], method="fdr_bh")
            for r, p_adj, rej in zip(pair_results, corrected[1], corrected[0]):
                r["p_fdr"]   = round(p_adj, 5)
                r["reject"]  = rej
                r["sig"]     = "***" if p_adj < .001 else "**" if p_adj < .01 else "*" if p_adj < .05 else "ns"
                r["higher"]  = r["g1"] if r["m1"] >= r["m2"] else r["g2"]
                pairwise_rows.append(r)

    return pd.DataFrame(global_rows), pd.DataFrame(pairwise_rows) if pairwise_rows else pd.DataFrame()


def print_group_stats(df, group_col, cols=OCEAN_COLS):
    """Print mean and SD per group and trait."""
    print(f"\n{'─'*55}\nMean & SD by {group_col}\n{'─'*55}")
    for grp, gdf in df.groupby(group_col, observed=True):
        print(f"\n  [{grp}]  n={len(gdf)}")
        for col in cols:
            v = gdf[col].dropna()
            print(f"    {col:25s}: mean={v.mean():.3f}  sd={v.std():.3f}")


### PLOTTING FUNCTIONS

def plot_ocean_distributions(df, cols=OCEAN_COLS, title="OCEAN Score Distributions", save_path=None):
    """
    Three-row panel per trait: boxplot (with human BFI baseline),
    histogram + KDE, and QQ-plot. Includes KS normality annotation.
    """
    n = len(cols)
    fig, axes = plt.subplots(
        3, n,
        figsize=(4 * n, 10),
        gridspec_kw={"height_ratios": [1, 1.2, 1.2]},
    )

    for i, col in enumerate(cols):
        data   = df[col].dropna()
        color  = IMPACT_SIX[i]
        z      = zscore(data)
        ks_stat, ks_p = kstest(z, "norm")
        ks_sig = "***" if ks_p < .001 else "**" if ks_p < .01 else "*" if ks_p < .05 else ""
        normality_txt = "non-normal" if ks_p < 0.05 else "approx. normal"

        ### Boxplot
        ax_box = axes[0, i]
        sns.boxplot(
            y=data, ax=ax_box, color=color,
            linecolor="black", fliersize=3, width=0.35,
            boxprops=dict(alpha=0.75), saturation=1,
        )
        if col in HUMAN_BFI_NORMS:
            ax_box.axhline(
                HUMAN_BFI_NORMS[col], linestyle="--",
                linewidth=1, color="#333333", alpha=0.85,
            )
        ax_box.set_ylim(1, 5)
        ax_box.set_xticks([])
        ax_box.set_ylabel("Score" if i == 0 else "")
        ax_box.set_title(col, fontweight="bold")
        ax_box.text(
            0.98, 0.95,
            f"M={data.mean():.2f}\nSD={data.std():.2f}",
            transform=ax_box.transAxes, ha="right", va="top", fontsize=17,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
        )
        ax_box.grid(axis="y", linestyle="--", alpha=0.3)

        ### Histogram + KDE
        ax_kde = axes[1, i]
        bins = int(np.sqrt(len(data)))
        sns.histplot(data, ax=ax_kde, bins=bins, stat="density",
                     color=color, alpha=0.4, edgecolor="white")
        sns.kdeplot(data, ax=ax_kde, color=color, fill=True,
                    alpha=0.2, linewidth=1.8, cut=0)
        ax_kde.text(
            0.05, 0.94,
            f"KS: p={ks_p:.3f}".replace("0.", ".") + f"{ks_sig}\n{normality_txt}",
            transform=ax_kde.transAxes, ha="left", va="top", fontsize=17,
            multialignment="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none"),
        )
        ax_kde.set_xlim(1, 5)
        ax_kde.set_ylim(0, 1.9)
        ax_kde.set_xticks([1, 2, 3, 4, 5])
        ax_kde.set_yticks([0.5, 1, 1.5])
        ax_kde.set_xlabel("Score")
        ax_kde.set_ylabel("Density" if i == 0 else "")
        ax_kde.grid(axis="x", linestyle="--", alpha=0.3)

        ### QQ-plot
        ax_qq = axes[2, i]
        qq = probplot(data, dist="norm")
        theoretical, ordered = qq[0]
        ax_qq.scatter(theoretical, ordered, s=26, alpha=0.75, color=color, linewidth=0.5)
        slope, intercept, _ = qq[1]
        xline = np.linspace(theoretical.min(), theoretical.max(), 100)
        ax_qq.plot(xline, slope * xline + intercept, color="#333333", linewidth=2, alpha=0.85)
        ax_qq.set_ylabel("Q-Q Plot" if i == 0 else "")
        ax_qq.set_title("")
        ax_qq.grid(alpha=0.25)

    fig.suptitle(title, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_binary_comparison(df, group_col, title, palette, order=None,
                           stats_df=None, cols=OCEAN_COLS, save_path=None, t_test=True):
    """
    Two-row panel (boxplot + KDE) comparing exactly two groups per trait.
    Optionally annotates with t-test results from bonferroni_ttest().
    """
    groups = [g for g in (order or sorted(df[group_col].dropna().unique()))
              if g in df[group_col].values]
    n   = len(cols)
    pal = palette[:len(groups)] if isinstance(palette, list) else sns.color_palette(palette, len(groups))
    cmap_dict = {g: pal[i] for i, g in enumerate(groups)}

    fig, axes = plt.subplots(
        2, n,
        figsize=(4 * n, 8),
        gridspec_kw={"height_ratios": [1.6, 1]},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(cols):
        ### Boxplot
        ax = axes[0, i]
        sns.boxplot(
            data=df, x=group_col, y=col, hue=group_col,
            order=groups, hue_order=groups, palette=pal, ax=ax,
            saturation=1, linewidth=1.0, fliersize=2, width=0.35,
            boxprops=dict(alpha=0.75),
            medianprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
        )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([str(g) for g in groups], rotation=25, ha="right")
        ax.set_ylim(1, 5)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if i == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_title(col, fontweight="bold")

        if t_test:
            if stats_df is not None and not stats_df.empty:
                row = stats_df[stats_df["trait"] == col]
                if not row.empty:
                    x_pos, y_pos = 0.08, 0.05
                    ha, va = "left", "bottom"
                    if col.lower() == "neuroticism":
                        x_pos, y_pos, va = 0.08, 0.95, "top"
                    sig = row["sig"].values[0]
                    sig_str = sig if sig != "ns" else ""
                    ax.text(
                        x_pos, y_pos,
                        f"t={row['t'].values[0]:.2f}\np={row['p_bonferroni'].values[0]:.3f}{sig_str}",
                        transform=ax.transAxes, ha="left", va=va, multialignment="left",
                        fontsize=13,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  alpha=0.65, edgecolor="none"),
                    )

        ### KDE
        ax2 = axes[1, i]
        for j, grp in enumerate(groups):
            d = df[df[group_col] == grp][col].dropna()
            sns.kdeplot(d, ax=ax2, label=str(grp), color=pal[j],
                        fill=True, alpha=0.25, linewidth=1.5, warn_singular=False)
        ax2.set_xlim(1, 5)
        ax2.set_xlabel(col)
        ax2.set_ylabel("Density" if i == 0 else "")
        ax2.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_multigroup_comparison(df, group_col, title, palette, order=None,
                               cols=OCEAN_COLS, save_path=None, figsize=None,
                               test=True, posthoc=True, kde=True):
    """
    Boxplot (+ optional KDE row) for ≥ 2 groups with Kruskal-Wallis /
    Mann-Whitney U annotation and pairwise FDR-BH posthoc labels.
    kde=True  → two-row layout (boxplot + KDE).
    kde=False → single-row boxplots only.
    test=False → suppress stat annotations.
    """
    groups = [g for g in (order or sorted(df[group_col].dropna().unique()))
              if g in df[group_col].values]
    n = len(cols)
    pal = (palette[:len(groups)] if isinstance(palette, list)
           else sns.color_palette(palette, len(groups)))

    global_stats, pairwise_stats = mannwhitney_pairwise(df, group_col, cols=cols, groups=groups)

    if kde:
        fig, axes = plt.subplots(
            2, n,
            figsize=figsize or (4 * n, 8),
            gridspec_kw={"height_ratios": [1.4, 1]},
        )
        if n == 1:
            axes = axes.reshape(2, 1)
        box_axes = axes[0]
        kde_axes = axes[1]
    else:
        fig, axes = plt.subplots(1, n, figsize=figsize or (4 * n, 3))
        if n == 1:
            axes = np.array([axes])
        box_axes = axes
        kde_axes = None

    for i, col in enumerate(cols):
        g_row = global_stats[global_stats["trait"] == col].iloc[0]
        p = g_row["p"]
        p_txt = f"p={p:.3f}" if p >= .001 else "p<.001"
        stars = g_row["sig"]

        posthoc_txt = ""
        if not pairwise_stats.empty:
            sig_pairs = pairwise_stats[
                (pairwise_stats["trait"] == col) & (pairwise_stats["reject"])
                ]
            lines = [
                f"{r['higher']} > {r['g2'] if r['higher'] == r['g1'] else r['g1']} {r['sig']}"
                for _, r in sig_pairs.iterrows()
            ]
            if lines:
                posthoc_txt = "\n" + "\n".join(lines)

        ### Boxplot
        ax = box_axes[i]
        sns.boxplot(
            data=df, x=group_col, y=col, hue=group_col,
            order=groups, hue_order=groups, palette=pal, legend=False,
            linewidth=1.1, width=0.42, fliersize=2.5, saturation=1,
            boxprops=dict(alpha=0.75),
            medianprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            ax=ax,
        )
        ax.set_ylim(1, 5)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if i == 0 else "")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=15)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_title(col, fontweight="bold")

        if test:
            stat_label = f"{g_row['stat_name']}={g_row['stat']:.2f}\n{p_txt} {stars}"
            if posthoc:
                stat_label = f"{g_row['stat_name']}={g_row['stat']:.2f}\n{p_txt} {stars}{posthoc_txt}"
            x_pos, y_pos = 0.08, 0.05
            ha, va = "left", "bottom"
            if col.lower() == "neuroticism":
                x_pos, y_pos, va = 0.08, 0.95, "top"
            ax.text(
                x_pos, y_pos, stat_label,
                transform=ax.transAxes, ha=ha, va=va, multialignment="left",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.7, edgecolor="none"),
            )

        ### KDE
        if kde:
            ax2 = kde_axes[i]
            for j, grp in enumerate(groups):
                d = df[df[group_col] == grp][col].dropna()
                sns.kdeplot(d, ax=ax2, color=pal[j], fill=True,
                            alpha=0.22, linewidth=2, warn_singular=False, label=str(grp))
            if i == 0:
                ax2.legend(title=group_col, loc="upper left",
                           frameon=True, fontsize=13, title_fontsize=13)
            ax2.set_xlim(1, 5)
            ax2.set_xlabel("")
            ax2.set_ylabel("Density" if i == 0 else "")
            ax2.grid(axis="x", linestyle="--", alpha=0.25)

    fig.suptitle(title, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig

def plot_trait_correlation_heatmap(corr_matrix, trait_labels_short, title="Trait Correlations",
                                   save_path=None):
    """
    Lower-triangle heatmap of a trait correlation / covariance matrix.
    `corr_matrix` is a square DataFrame with full trait names as index/columns.
    `trait_labels_short` maps full → short label for display.
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    display = corr_matrix.copy()
    display.index   = [trait_labels_short.get(t, t) for t in corr_matrix.index]
    display.columns = [trait_labels_short.get(t, t) for t in corr_matrix.columns]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=False)
    sns.heatmap(
        display, mask=mask, annot=True, fmt=".2f", cmap=DIVERGING_CMAP,
        center=0, vmin=-1.2, vmax=1.2, square=True,
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Std. covariance", "fraction": 0.05, "pad": 0.02},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_family_heatmap(df, group_col, order, cols=OCEAN_COLS,
                        title="OCEAN Trait Profiles by Model Family",
                        save_path=None, figsize=(9, 8)):
    """
    Z-scored mean OCEAN heatmap grouped by model family.
    """

    TRAIT_ORDER = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ]

    TRAIT_LABELS = {
        "Openness": "O",
        "Conscientiousness": "C",
        "Extraversion": "E",
        "Agreeableness": "A",
        "Neuroticism": "N",
    }

    heat = df.groupby(group_col)[TRAIT_ORDER].mean().loc[order]
    heat_z = (heat - heat.mean()) / heat.std()
    heat_z = heat_z.rename(columns=TRAIT_LABELS)


    # Clean display names
    heat_z.index = [FAMILY_LABELS.get(idx, idx.title()) for idx in heat_z.index]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heat_z,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_CMAP,
        linewidths=0.5,
        cbar_kws={"label": "z-score"},
        ax=ax,
        vmin=-2.2,
        vmax=2.2,
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    return fig


def plot_release_date_regression(df, date_col, release_months_col, palette, figsize,
                                 cols=OCEAN_COLS, title="OCEAN Scores over Release Date",
                                 save_path=None, test=True):
    """
    Scatter + OLS regression line (± 95% CI) of each trait against release date.
    """
    import matplotlib.dates as mdates
    import scipy.stats as scipy_stats

    ref = pd.Timestamp("2026-05-12")

    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # axes = axes.flatten()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 6)  # finer grid

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),  # centered left
        fig.add_subplot(gs[1, 3:5]),  # centered right
    ]

    for i, col in enumerate(cols):
        ax  = axes[i]
        tmp = df[[date_col, release_months_col, col]].dropna()
        x   = tmp[release_months_col].values
        y   = tmp[col].values

        slope, intercept, r, p, se = scipy_stats.linregress(x, y)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

        x_line    = np.linspace(x.min(), x.max(), 100)
        y_line    = slope * x_line + intercept
        dates_line = pd.to_datetime(ref + pd.to_timedelta(x_line * 30.44, unit="D"))

        n_obs  = len(x)
        t_val  = scipy_stats.t.ppf(0.975, df=n_obs - 2)
        s_err  = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n_obs - 2))
        ci     = t_val * s_err * np.sqrt(
            1 / n_obs + (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
        )

        label = None

        if test:
            label = f"β={slope:.2f}, p={p:.3f}{star}"

        ax.scatter(tmp[date_col], y, color=palette[0], alpha=0.25, s=85, zorder=3)
        ax.plot(dates_line, y_line, color=palette[1], linewidth=3,
                label=label)
        ax.fill_between(dates_line, y_line - ci, y_line + ci,
                        color=palette[1], alpha=0.08)
        ax.set_ylim(1, 5)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Release date")
        ax.set_ylabel("Score" if i % 3 == 0 else "")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=18)
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend()

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_param_scale_regression(df, params_col, palette, figsize, title,
                                cols=OCEAN_COLS,
                                save_path=None, test=True):
    """
    Scatter + OLS regression (± 95% CI) on log10(params) scale per trait.
    """
    import scipy.stats as scipy_stats

    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # axes = axes.flatten()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 6)  # finer grid

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),  # centered left
        fig.add_subplot(gs[1, 3:5]),  # centered right
    ]

    df = df.copy()
    df[params_col] = pd.to_numeric(df[params_col], errors="coerce")
    tmp = df[df[params_col] > 0].dropna(subset=[params_col] + cols)

    for i, col in enumerate(cols):
        ax = axes[i]
        x = np.log10(tmp[params_col].values)
        y = tmp[col].values

        slope, intercept, r, p, se = scipy_stats.linregress(x, y)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ci = 1.96 * se * np.sqrt(
            1 + 1 / len(x) + (x_line - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
        )

        label = None

        if test:
            label = f"β={slope:.2f}, p={p:.3f}{star}"

        ax.scatter(tmp[params_col], y, color=palette[0], alpha=0.45, s=85, zorder=3)
        ax.plot(10 ** x_line, y_line, color=palette[1], linewidth=3,
                label=label)
        ax.fill_between(10 ** x_line, y_line - ci, y_line + ci,
                        color=palette[1], alpha=0.12)
        ax.set_xscale("log")
        ax.set_ylim(1, 5)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Parameters (B, log scale)")
        ax.set_ylabel("Score" if i % 3 == 0 else "")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}B"))
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend()

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_trait_covariances(
    cov_values,
    traits_full = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ],
    # traits_full=("Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"),
    trait_labels = ["O", "C", "E", "A", "N"],
    # trait_labels=("E", "A", "C", "N", "O"),
    title="Trait Covariances",
    save_path=None,
    vmin=-1.0,
    vmax=1.0,
):
    """
    Plot symmetric covariance/correlation matrix from lavaan-style output.

    cov_values: dict with keys (trait_i, trait_j) -> value
    """

    # -----------------------------
    # build matrix
    # -----------------------------
    corr = pd.DataFrame(
        np.eye(len(traits_full)),
        index=list(traits_full),
        columns=list(traits_full),
    )

    for (a, b), v in cov_values.items():
        corr.loc[a, b] = v
        corr.loc[b, a] = v

    # reorder for display
    corr_plot = corr.copy()
    corr_plot.index = trait_labels
    corr_plot.columns = trait_labels

    # -----------------------------
    # mask upper triangle
    # -----------------------------
    mask = np.triu(np.ones_like(corr_plot, dtype=bool), k=1)

    # -----------------------------
    # colormap (your paper style)
    # -----------------------------
    impact_six = ["#008CBB", "#9569D1", "#A3D900", "#E30053", "#FFB000", "#3092FF"]

    cmap = LinearSegmentedColormap.from_list(
        "impact_custom",
        [impact_six[3], "#FFFFFF", impact_six[4]],
        N=256
    )

    # -----------------------------
    # plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=False)

    sns.heatmap(
        corr_plot,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={
            "label": "Std. covariance",
            "fraction": 0.05,
            "pad": 0.02
        }
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig
