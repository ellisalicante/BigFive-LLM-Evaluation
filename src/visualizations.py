### IMPORTS
import io, os, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as scipy_stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests


### CONFIG
IMPACT_SIX = ["#008CBB", "#9569D1", "#A3D900", "#E30053", "#FFB000", "#3092FF"]
BINARY_PAL_BLUE_PURPLE = ["#008CBB", "#9569D1"]
BINARY_PAL_GREEN_RED = ["#A3D900", "#E30053"]
BINARY_PAL_BLUE_ORANGE = ["#008CBB", "#FFB000"]
BINARY_PAL_BLUE_PINK = ["#008CBB", "#E30053"]
TRI_COLOR = ["#E30053", "#FFB000", "#008CBB"]
GIF_COLORS = ["#D95F5F", "#5B8DD9", "#E0A040", "#5BAD72", "#9B6DD9", "#3DBDBD", "#D96BB0", "#B0B040",
              "#40BF96", "#7040D9", "#D97840", "#40A0D9", "#D94040", "#4060D9", "#B05BA0", "#60B060"]

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "impact_diverging",
    [IMPACT_SIX[3], "#FFFFFF", IMPACT_SIX[0]],
    N=256,
)

OCEAN_COLS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

OCEAN_LABELS = ["O", "C", "E", "A", "N"]

TRAIT_LABELS = {
    "Openness": "O",
    "Conscientiousness": "C",
    "Extraversion": "E",
    "Agreeableness": "A",
    "Neuroticism": "N",
}

HUMAN_BFI_NORMS = {
    "Conscientiousness": 3.74,
    "Agreeableness":     3.95,
    "Neuroticism":       2.85,
    "Openness":          3.24,
    "Extraversion":      3.53,
}

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
    "falcon": "Falcon",
    "granite": "Granite",
    "olmo": "Olmo",
}

FIG_WIDTH  = 7.0
FIG_HEIGHT = 3.5
SCALE_MIN = 1.0
SCALE_MAX = 5.0



# HELPERS
def apply_paper_style():
    plt.rcParams.update({
        "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.titlepad": 10,
        "axes.labelpad": 10,
    })


# DESCRIPTIVES
def descriptive_stats(df,
                      cols):

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


def sample_desc(df_all_raw,
                df_metadata,
                columns):

    n_models_collected = df_all_raw["model"].nunique()
    n_models_valid = df_metadata["model"].nunique()

    families = (
        df_metadata["Family"]
        .dropna()
        .astype(str)
        .str.lower()
        .nunique()
    )

    families_vals = (
        df_metadata["Family"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    regions = (
        df_metadata["Region"]
        .dropna()
        .value_counts()
    )

    country_family = (
        df_metadata[["Region", "Family"]]
        .dropna()
        .groupby("Region")["Family"]
        .apply(lambda x: x.value_counts().index.tolist())
    )

    licenses = (
        df_metadata["license_group"]
        .value_counts(dropna=False)
    )

    reasoning_counts = (
        df_metadata["Reasoning"]
        .astype(str)
        .str.lower()
        .value_counts()
    )

    release_dates = pd.to_datetime(
        df_metadata["Release_date"],
        errors="coerce"
    )

    min_release = release_dates.min()
    max_release = release_dates.max()

    params = pd.to_numeric(
        df_metadata["params_numeric"],
        errors="coerce"
    )

    min_params = params.min()
    max_params = params.max()

    df_all_ocean = df_all_raw[df_all_raw["dimension"].isin(columns)]

    valid_mask_ocean = pd.to_numeric(
        df_all_ocean["response"],
        errors="coerce"
    ).isin([1, 2, 3, 4, 5])

    total_rows_ocean = len(df_all_ocean)

    n_valid_ocean = valid_mask_ocean.sum()
    n_invalid_ocean = (~valid_mask_ocean).sum()

    completion_rate_ocean = n_valid_ocean / total_rows_ocean * 100
    refusal_rate_ocean = n_invalid_ocean / total_rows_ocean * 100

    model_refusal = (
        df_all_ocean.assign(
            valid=pd.to_numeric(
                df_all_ocean["response"],
                errors="coerce"
            ).isin([1, 2, 3, 4, 5])
        )
        .groupby("model")
        .agg(
            total=("valid", "size"),
            valid=("valid", "sum")
        )
    )

    model_refusal["refusal_rate"] = (1 - (model_refusal["valid"] / model_refusal["total"])) * 100

    ocean_means = descriptive_stats(df_metadata, cols=columns)

    stats = {
        "n_models_collected": n_models_collected,
        "n_models_valid": n_models_valid,
        "families": families,
        "families_vals": families_vals,
        "regions": regions,
        "country_family": country_family,
        "licenses": licenses,
        "reasoning_counts": reasoning_counts,
        "min_release": min_release,
        "max_release": max_release,
        "min_params": min_params,
        "max_params": max_params,
        "total_rows_OCEAN": total_rows_ocean,
        "n_valid_OCEAN": n_valid_ocean,
        "n_invalid_OCEAN": n_invalid_ocean,
        "completion_rate_OCEAN": completion_rate_ocean,
        "refusal_rate_OCEAN": refusal_rate_ocean,
        "model_refusal": model_refusal
    }

    return stats, ocean_means



# TESTS

def ks_normality_tests(df,
                       cols):

    rows = []

    for col in cols:
        data = df[col].dropna()
        z = scipy_stats.zscore(data)
        ks_stat, ks_p = scipy_stats.kstest(z, "norm")
        rows.append({
            "trait":          col,
            "D":              round(ks_stat, 3),
            "p":              round(ks_p, 3),
            "interpretation": "non-normal" if ks_p < 0.05 else "approx. normal",
        })

    return pd.DataFrame(rows)


def bonferroni_ttest(df,
                     group_col,
                     cols):

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
        t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
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


def paired_ttest_base_vs_it(df_models,
                            df_metadata,
                            pairs,
                            cols):

    results = []

    for base, inst in pairs:
        base_row = df_models[df_models["model"] == base]
        inst_row = df_models[df_models["model"] == inst]
        params = df_metadata[df_metadata["model"] == inst]["Parameters_B"]
        if base_row.empty or inst_row.empty:
            print(f"Missing pair: {base} vs {inst}")
            continue
        base_row = base_row.iloc[0]
        inst_row = inst_row.iloc[0]
        for trait in cols:
            results.append({
                "pair":        f"{base[:30]} → {inst[:30]}",
                "params":      params,
                "trait":       trait,
                "base":        base_row[trait],
                "instruction": inst_row[trait],
                "diff":        inst_row[trait] - base_row[trait],
            })

    df_pairs = pd.DataFrame(results)

    stats_rows = []

    for trait in cols:
        d = df_pairs[df_pairs["trait"] == trait]["diff"].dropna()
        t, p = scipy_stats.ttest_1samp(d, 0)
        stats_rows.append({
            "trait":     trait,
            "mean_diff": round(d.mean(), 2),
            "sd_diff":   round(d.std(), 2),
            "t":         round(t, 3),
            "p":         round(p, 5),
        })

    print("\n\n=== Per-pair OCEAN differences (IT − Base) ===\n")

    pair_table = (
        df_pairs
        .pivot(index="pair", columns="trait", values="diff")
        .round(3)
    )

    pair_table = pair_table[
        OCEAN_COLS
    ]
    pair_table.columns = OCEAN_LABELS

    print(pair_table.to_string())

    print(f"\nN pairs = {len(pair_table)}")

    return df_pairs, pd.DataFrame(stats_rows)


def mannwhitney_pairwise(df,
                         group_col,
                         cols,
                         groups=None):

    if groups is None:
        groups = sorted(df[group_col].dropna().unique())

    global_rows = []
    pairwise_rows = []

    for col in cols:
        samples = [df[df[group_col] == g][col].dropna() for g in groups]

        if len(groups) == 2:
            stat, p = scipy_stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
            stat_name = "U"
        else:
            stat, p = scipy_stats.kruskal(*samples)
            stat_name = "H"

        global_rows.append({
            "trait":     col,
            "stat_name": stat_name,
            "stat":      round(stat, 3),
            "p":         round(p, 5),
            "sig":       "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "",
        })

        if len(groups) > 2 and p < .05:
            pair_results = []
            for g1, g2 in itertools.combinations(groups, 2):
                x = df[df[group_col] == g1][col].dropna()
                y = df[df[group_col] == g2][col].dropna()
                u_pair, p_pair = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
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


def print_group_stats(df,
                      group_col):

    print(f"\n{'─'*55}\nMean & SD by {group_col}\n{'─'*55}")

    for grp, gdf in df.groupby(group_col, observed=True):
        print(f"\n  [{grp}]  n={len(gdf)}")
        for col in OCEAN_COLS:
            v = gdf[col].dropna()
            print(f"    {col:25s}: mean={v.mean():.3f}  sd={v.std():.3f}")



### PLOTTING FUNCTIONS
def plot_ocean_distributions(df,
                             cols,
                             title=None,
                             save_path=None,
                             show_mean=True):

    n = len(cols)
    fig, axes = plt.subplots(
        2, n,
        figsize=(4 * n, 6),
    )

    print("N =", len(df))

    for i, col in enumerate(cols):
        data = df[col].dropna()
        color = IMPACT_SIX[i]
        z = scipy_stats.zscore(data)
        ks_stat, ks_p = scipy_stats.kstest(z, "norm")
        ks_sig = "***" if ks_p < .001 else "**" if ks_p < .01 else "*" if ks_p < .05 else ""
        normality_txt = "non-normal" if ks_p < 0.05 else "approx. normal"

        # Boxplot
        ax_box = axes[0, i]
        sns.boxplot(
            y=data, ax=ax_box, color=color,
            linecolor="black", fliersize=3, width=0.35,
            boxprops=dict(alpha=0.85), saturation=1,
        )
        if show_mean:
            if col in HUMAN_BFI_NORMS:
                ax_box.axhline(
                    HUMAN_BFI_NORMS[col], linestyle="--",
                    linewidth=1, color="#333333", alpha=0.85,
                )
        ax_box.set_ylim(1, 5)
        ax_box.set_yticks([1, 2, 3, 4, 5])
        ax_box.set_xticks([])
        ax_box.set_ylabel("Score" if i == 0 else "")
        ax_box.set_title(col, fontweight="bold", pad=20)
        ax_box.tick_params(axis="y", pad=8, length=0)
        ax_box.tick_params(
            axis="x",
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
            pad=8,
            length=0
        )
        ax_box.set_xlabel("")
        ax_box.text(
            0.05, 0.05,
            f"M={data.mean():.2f}\nSD={data.std():.2f}",
            transform=ax_box.transAxes, ha="left", va="bottom", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
        )
        ax_box.grid(axis="y", linestyle="--", alpha=0.35)

        # Histogram + KDE
        ax_kde = axes[1, i]
        bins = 15
        sns.histplot(data, ax=ax_kde, bins=bins, stat="density",
                     color=color, alpha=0.6, edgecolor="white")
        sns.kdeplot(data, ax=ax_kde, color=color, fill=True,
                    alpha=0.2, linewidth=1.8, cut=0)
        ax_kde.text(
            0.05, 0.94,
            f"KS: p={ks_p:.3f}".replace("0.", ".") + f"{ks_sig}\n{normality_txt}",
            transform=ax_kde.transAxes, ha="left", va="top", fontsize=13,
            multialignment="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax_kde.set_xlim(1, 5)
        ax_kde.set_ylim(0, 1.75)
        ax_kde.set_xticks([1, 2, 3, 4, 5])
        ax_kde.set_yticks([1])
        ax_kde.tick_params(axis="y", pad=8, length=0)
        ax_kde.tick_params(axis="x", pad=8, length=0)
        ax_kde.set_xlabel("Score")
        ax_kde.set_ylabel("Density" if i == 0 else "")
        ax_kde.grid(axis="x", linestyle="--", alpha=0.35)

        # # QQ-plot
        # ax_qq = axes[2, i]
        # qq = probplot(data, dist="norm")
        # theoretical, ordered = qq[0]
        # ax_qq.scatter(theoretical, ordered, s=26, alpha=0.75, color=color, linewidth=0.5)
        # slope, intercept, _ = qq[1]
        # xline = np.linspace(theoretical.min(), theoretical.max(), 100)
        # ax_qq.plot(xline, slope * xline + intercept, color="#333333", linewidth=2, alpha=0.85)
        # ax_qq.set_ylabel("Q-Q Plot" if i == 0 else "")
        # # ax_qq.set_title("")
        # ax_qq.set_ylim(1, 5)
        # ax_qq.tick_params(axis="y", pad=8, length=0)
        # ax_qq.tick_params(axis="x", pad=8, length=0)
        # ax_qq.grid(alpha=0.35)

    fig.suptitle(title, fontweight="bold", fontsize=19)
    fig.tight_layout()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_binary_comparison(df,
                           group_col,
                           title,
                           palette,
                           order=None,
                           display_labels=None,
                           stats_df=None,
                           cols=None,
                           save_path=None,
                           t_test=True):

    groups = [g for g in (order or sorted(df[group_col].dropna().unique()))
              if g in df[group_col].values]
    labels = [display_labels.get(g, str(g)) if display_labels else str(g)
              for g in groups]
    n   = len(cols)
    pal = palette[:len(groups)] if isinstance(palette, list) else sns.color_palette(palette, len(groups))

    fig, axes = plt.subplots(
        2, n,
        figsize=(4 * n, 6),
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(cols):

        # Boxplot
        ax = axes[0, i]
        sns.boxplot(
            data=df,
            x=group_col,
            y=col,
            hue=group_col,
            palette=pal,
            ax=ax,
            saturation=1,
            linewidth=1.0,
            fliersize=2,
            width=0.35,
            boxprops=dict(alpha=0.9),
            medianprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
        )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=13)
        ax.set_ylim(1, 5)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if i == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_title(col, fontweight="bold", pad=20)

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

        # KDE
        ax2 = axes[1, i]
        for j, grp in enumerate(groups):
            d = df[df[group_col] == grp][col].dropna()
            sns.kdeplot(d, ax=ax2, label=labels[j], color=pal[j],
                        fill=True, alpha=0.35, linewidth=1.5, warn_singular=False)
        if i == 0:
            ax2.legend(
                loc="upper left",
                frameon=False,
                fontsize=13,
                labelspacing=0.2,
                borderpad=0.2,
                handletextpad=0.4,
            )
        ax2.set_xlim(1, 5)
        ax2.set_ylim(0, 1.6)
        ax2.set_xlabel("")
        ax2.set_ylabel("Density" if i == 0 else "")
        ax2.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=19)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_multigroup_comparison(df,
                               group_col,
                               title,
                               palette,
                               order=None,
                               cols=None,
                               save_path=None,
                               figsize=None,
                               test=True,
                               posthoc=True,
                               kde=True):

    groups = [g for g in (order or sorted(df[group_col].dropna().unique()))
              if g in df[group_col].values]
    n = len(cols)
    pal = (palette[:len(groups)] if isinstance(palette, list)
           else sns.color_palette(palette, len(groups)))

    global_stats, pairwise_stats = mannwhitney_pairwise(df, group_col, cols=cols, groups=groups)

    if kde:
        fig, axes = plt.subplots(
            2, n,
            figsize=figsize or (4 * n, 7),
            gridspec_kw={"height_ratios": [1.2, 1]},
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
        kde_axes = np.array([])

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

        # Boxplot
        ax = box_axes[i]
        sns.boxplot(
            data=df,
            x=group_col,
            y=col,
            hue=group_col,
            palette=pal,
            legend=False,
            linewidth=1.1,
            width=0.42,
            fliersize=2.5,
            saturation=1,
            boxprops=dict(alpha=0.9),
            medianprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            ax=ax,
        )
        ax.set_ylim(1, 5)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if i == 0 else "")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_title(col, fontweight="bold", pad=20)

        if test:
            stat_label = f"{g_row['stat_name']}={g_row['stat']:.2f}\n{p_txt} {stars}"
            if posthoc:
                stat_label = f"{g_row['stat_name']}={g_row['stat']:.2f}\n{p_txt} {stars}{posthoc_txt}"
            x_pos, y_pos = 0.08, 0.05
            ha, va = "left", "bottom"
            if col.lower() == "neuroticism":
                x_pos, y_pos, va = 0.08, 0.95, "top"
            ax.text(
                x_pos,
                y_pos,
                stat_label,
                transform=ax.transAxes,
                ha=ha,
                va=va,
                multialignment="left",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.7, edgecolor="none"),
            )

        # KDE
        if kde:
            ax2 = kde_axes[i]
            for j, grp in enumerate(groups):
                d = df[df[group_col] == grp][col].dropna()
                sns.kdeplot(d,
                            ax=ax2,
                            color=pal[j],
                            fill=True,
                            alpha=0.22,
                            linewidth=2,
                            warn_singular=False,
                            label=str(grp))
            if i == 0:
                ax2.legend(
                    loc="upper left",
                    frameon=False,
                    fontsize=13,
                    labelspacing=0.2,
                    borderpad=0.2,
                    handletextpad=0.4,
                )
            ax2.set_xlim(1, 5)
            ax2.set_ylim(0, 1.9)
            ax2.set_xlabel("")
            ax2.set_ylabel("Density" if i == 0 else "")
            ax2.grid(axis="x", linestyle="--", alpha=0.25)

    fig.suptitle(title, fontweight="bold", fontsize=19)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_family_heatmap_percentile(df,
                                   group_col,
                                   order,
                                   title=None,
                                   save_path=None,
                                   figsize=(9, 8)):

    heat = df.groupby(group_col)[OCEAN_COLS].mean().loc[order]
    heat_pct = heat.rank(pct=True) * 100
    heat_pct = heat_pct.rename(columns=TRAIT_LABELS)

    counts = df.groupby(group_col).size().loc[order]

    heat_pct.index = [
        f"{FAMILY_LABELS.get(idx, idx.title())} ({counts.loc[idx]})"
        for idx in heat_pct.index
    ]

    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list(
        "impact_custom",
        [
            (0.00, "#BD0045"),
            (0.05, IMPACT_SIX[3]),
            (0.50, "#FFFFFF"),
            (0.95, IMPACT_SIX[0]),
            (1.00, "#007096"),
        ],
        N=256,
    )

    sns.heatmap(
        heat_pct,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": "Percentile rank"},
        ax=ax,
        vmin=0,
        vmax=100,
        center=50,
    )

    ax.set_title(title, fontweight="bold", pad=25)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        ha="center",
        fontweight="bold",
    )

    ax.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        pad=8,
        length=0
    )

    ax.tick_params(axis="y", pad=8, length=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    return fig


def plot_regression(df,
                    x_type,
                    date_col=None,
                    release_months_col=None,
                    params_col=None,
                    palette=None,
                    figsize=None,
                    cols=None,
                    title=None,
                    save_path=None,
                    test=True):

    if x_type not in ("release_date", "param_scale"):
        raise ValueError(f"Invalid x_type: {x_type}")

    ref = pd.Timestamp("2026-05-12")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 6)

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    tmp_all = pd.DataFrame()
    if x_type == "param_scale":
        df = df.copy()
        df[params_col] = pd.to_numeric(df[params_col], errors="coerce")
        tmp_all = df[df[params_col] > 0].dropna(subset=[params_col] + cols)

    for i, col in enumerate(cols):
        ax = axes[i]

        if x_type == "release_date":
            tmp = df[[date_col, release_months_col, col]].dropna()
            x = tmp[release_months_col].values
            y = tmp[col].values
        else:
            tmp = tmp_all
            x = np.log10(tmp[params_col].values)
            y = tmp[col].values

        slope, intercept, r, p, se = scipy_stats.linregress(x, y)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        dates_line = pd.DatetimeIndex([])
        if x_type == "release_date":
            dates_line = pd.to_datetime(ref + pd.to_timedelta(x_line * 30.44, unit="D"))
            n_obs = len(x)
            t_val = scipy_stats.t.ppf(0.975, df=n_obs - 2)
            s_err = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n_obs - 2))
            ci = t_val * s_err * np.sqrt(
                1 / n_obs + (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
            )
        else:
            ci = 1.96 * se * np.sqrt(
                1 + 1 / len(x) + (x_line - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
            )

        label = None

        if test:
            label = f"β={slope:.2f}, p={p:.3f}{star}"

        if x_type == "release_date":
            ax.scatter(tmp[date_col],
                       y,
                       color=palette[0],
                       alpha=0.3,
                       s=85,
                       zorder=3)
            ax.plot(dates_line,
                    y_line,
                    color=palette[1],
                    linewidth=3,
                    label=label)
            ax.fill_between(dates_line,
                            y_line - ci,
                            y_line + ci,
                            color=palette[1],
                            alpha=0.08)
        else:
            ax.scatter(tmp[params_col],
                       y,
                       color=palette[0],
                       alpha=0.4,
                       s=85,
                       zorder=3)
            ax.plot(10 ** x_line,
                    y_line,
                    color=palette[1],
                    linewidth=3,
                    label=label)
            ax.fill_between(10 ** x_line,
                            y_line - ci,
                            y_line + ci,
                            color=palette[1],
                            alpha=0.15)

        ax.set_ylim(1, 5)
        ax.set_title(col, fontweight="bold", pad=15)
        ax.set_ylabel("Score" if i % 3 == 0 else "")

        if x_type == "release_date":
            ax.set_xlabel("")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=18)
        else:
            ax.set_xscale("log")
            ax.set_xlabel("Parameters (B, log scale)")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x_xax, _: f"{x_xax:g}"))

        ax.grid(linestyle="--", alpha=0.3)

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontweight="bold", fontsize=19)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=False)
    return fig


def plot_trait_covariances(cov_values,
                           title="Trait Covariances",
                           save_path=None,
                           vmin=-1.0,
                           vmax=1.0):

    corr = pd.DataFrame(
        np.eye(len(OCEAN_COLS)),
        index=list(OCEAN_COLS),
        columns=list(OCEAN_COLS),
    )

    for (a, b), v in cov_values.items():
        corr.loc[a, b] = v
        corr.loc[b, a] = v

    corr_plot = corr.copy()
    corr_plot.index = OCEAN_LABELS
    corr_plot.columns = OCEAN_LABELS

    mask = np.triu(np.ones_like(corr_plot, dtype=bool), k=1)

    cmap = LinearSegmentedColormap.from_list(
        "impact_custom",
        [
            (0.00, "#BD0045"),
            (0.05, IMPACT_SIX[3]),
            (0.49, "#FFFFFF"),
            (0.51, "#FFFFFF"),
            (0.98, IMPACT_SIX[0]),
            (1.00, "#007096"),
        ],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=False)

    sns.heatmap(
        corr_plot,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={
            "label": "Standardized covariance",
            "fraction": 0.05,
            "pad": 0.02
        }
    )

    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="x", pad=8, length=0)
    ax.tick_params(axis="y", pad=8, length=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def _norm(v):
    return (v - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)


def close(arr):
    return list(arr) + [arr[0]]


def _draw_radar(ax,
                means,
                global_means,
                color,
                trait_names,
                show_labels=True,
                show_scores=True,
                title=None):

    n = len(trait_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    mn  = [_norm(v) for v in means]
    ang = close(angles)
    mnc = close(mn)

    ax.set_facecolor("none")
    ax.patch.set_visible(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid rings
    theta_full = np.linspace(0, 2 * np.pi, 300)
    ring_scores = [2, 3, 4, 5]
    for rs in ring_scores:
        rv = _norm(rs)
        ax.plot(theta_full,
                [rv] * 300,
                color="#E0E0E0",
                lw=0.55,
                zorder=1)
        if show_labels:
            ax.text(np.pi / 2,
                    rv + 0.025,
                    str(rs),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#BBBBBB",
                    fontfamily="monospace")

    # Spokes
    for ang_i in angles:
        ax.plot([ang_i, ang_i],
                [0, 1.0],
                color="#E8E8E8",
                lw=0.7,
                zorder=1)

    # SD band
    # if sds is not None:
    #     up = [_norm(min(m + s, SCALE_MAX)) for m, s in zip(means, sds)]
    #     lo = [_norm(max(m - s, SCALE_MIN)) for m, s in zip(means, sds)]
    #     ax.fill_between(close(angles), close(lo), close(up),
    #                     color=color, alpha=0.13, zorder=2, linewidth=0)

    # Global mean polygon (all models)
    global_mn = [_norm(v) for v in global_means]
    ax.plot(close(angles),
            close(global_mn),
            color="#AAAAAA",
            lw=1.2,
            zorder=3,
            linestyle="--",
            dash_capstyle="round",
            alpha=0.5)
    ax.scatter(angles,
               global_mn,
               s=18,
               color="#AAAAAA",
               zorder=5,
               edgecolors="white",
               linewidths=1.0)

    # Filled polygon
    ax.fill(ang, mnc, color=color, alpha=0.2, zorder=2)
    ax.plot(ang, mnc, color=color, lw=2.0, zorder=3, solid_capstyle="round")

    # Outer halo dots
    ax.scatter(angles,
               mn,
               s=170,
               color=color,
               alpha=0.18,
               edgecolors="none",
               zorder=4)

    # Vertex dots
    ax.scatter(angles,
               mn,
               s=65,
               color=color,
               zorder=5,
               edgecolors="white",
               linewidths=1.6)

    # Score labels
    if show_scores:
        for ang_i, mn_i, mv in zip(angles, mn, means):
            offset = 0.14
            r_lbl = mn_i + offset if mn_i < 0.80 else mn_i - offset
            va = "bottom" if mn_i < 0.80 else "top"
            ax.text(ang_i,
                    r_lbl,
                    f"{mv:.2f}",
                    ha="center",
                    va=va,
                    fontsize=7.5,
                    color=color,
                    fontweight="bold",
                    fontfamily="monospace",
                    zorder=6)

    # Trait axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    if show_labels:

        label_r = {
            "Openness": 1.10,
            "Conscientiousness": 1.40,
            "Extraversion": 1.25,
            "Agreeableness": 1.25,
            "Neuroticism": 1.30,
        }

        for ang_i, trait in zip(angles, trait_names):
            r = label_r.get(trait, 1.30)
            ax.text(ang_i,
                    r,
                    trait,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#2A2A2A",
                    fontweight="bold",
                    fontfamily="monospace",
                    zorder=7)

    ax.set_ylim(0, 1.46)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, fontfamily="monospace", pad=28)


def make_individual_png(df_family, family_name, means, global_means, color, trait_names,
                        display_name=None, dpi=150):
    fig = plt.figure(figsize=(5.6, 5.6), dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111, projection="polar", facecolor="white")

    _draw_radar(ax, means, global_means, color, trait_names,
                show_labels=True, show_scores=True)

    label = display_name or family_name
    fig.text(0.5,
             0.97,
             label,
             ha="center",
             va="top",
             fontsize=17,
             fontweight="bold",
             color=color,
             fontfamily="monospace")
    fig.text(0.5,
             0.92,
             "Big Five Profile",
             ha="center",
             va="top",
             fontsize=8.5,
             color="#AAAAAA",
             fontfamily="monospace")
    n_models = df_family[df_family["FamilyGrouped"] == family_name].shape[0]  # pass n_models as arg
    fig.text(0.5,
             0.88,
             f"N = {n_models}",
             ha="center",
             va="top",
             fontsize=8,
             color="#AAAAAA",
             fontfamily="monospace",
             fontstyle="italic")
    return fig


def make_all_plots(df_family,
                   family_order,
                   trait_order,
                   family_labels=None,
                   out_dir="ocean_frames",
                   gif_path="ocean_families.gif",
                   gif_duration_ms=1400,
                   dpi_individual=300):

    os.makedirs(out_dir, exist_ok=True)

    heat = (df_family.groupby("FamilyGrouped")[trait_order]
            .mean()
            .loc[family_order])
    means_dict = {fam: heat.loc[fam].values for fam in family_order}
    display_names = family_labels or {}

    gif_frames = []
    png_paths  = []

    global_means = df_family[trait_order].mean().values  # shape (n_traits,)

    for idx, fam in enumerate(family_order):
        color = GIF_COLORS[idx % len(GIF_COLORS)]
        label = display_names.get(fam, fam)

        fig = make_individual_png(
            df_family=df_family,
            family_name=fam,
            means=means_dict[fam],
            global_means=global_means,
            color=color,
            trait_names=trait_order,
            display_name=label,
            dpi=dpi_individual,
        )

        png_path = os.path.join(out_dir, f"{idx:02d}_{fam}.png")
        fig.savefig(png_path,
                    dpi=dpi_individual,
                    bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        png_paths.append(png_path)

        buf = io.BytesIO()
        fig2 = make_individual_png(
            df_family=df_family,
            family_name=fam,
            means=means_dict[fam],
            global_means=global_means,
            color=color,
            trait_names=trait_order,
            display_name=label,
            dpi=300,
        )
        fig2.savefig(buf,
                     format="png",
                     dpi=300,
                     bbox_inches="tight",
                     facecolor="white")
        plt.close(fig2)
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())

    # GIF
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=gif_duration_ms,
        loop=0,
        optimize=False,
    )

    print(f"\nDone — {len(family_order)} PNGs + GIF")