### 0) IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from src.visualizations import (
    IMPACT_SIX, IMPACT_TEN, DIVERGING_CMAP,
    OCEAN_COLS, apply_paper_style,
)


### PALETTES

CLUSTER_PAL = [
    "#008CBB", "#E30053", "#A3D900", "#FFB000",
    "#9569D1", "#3092FF", "#00BFA5", "#FF6F61",
]


### PREPARATION HELPERS

def prepare_item_matrix(df_cfa):
    """
    Drop models with any missing items, z-score columns.
    Returns (X_scaled, model_labels).
    """
    df = df_cfa.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    return X, df.index.tolist()


def prepare_trait_matrix(df_B, cols=OCEAN_COLS):
    """
    Drop models with any missing trait score, z-score.
    Returns (X_scaled, model_labels).
    """
    df = df_B.dropna(subset=cols).set_index("model")[cols]
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    return X, df.index.tolist()


### PCA HELPERS

def run_pca(X, n_components=None):
    """
    Fit PCA. Returns (pca, X_pca, explained_variance_ratio).
    n_components=None keeps all components.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def pca_loadings(pca, feature_names, n_components=5):
    """Return DataFrame of PCA loadings (features × components)."""
    n = min(n_components, pca.n_components_)
    cols = [f"PC{i+1}" for i in range(n)]
    return pd.DataFrame(
        pca.components_[:n].T,
        index=feature_names,
        columns=cols,
    )


### T-SNE HELPERS

def run_tsne(X, perplexity=30, n_iter=1000, random_state=42):
    """Fit t-SNE on X. Returns 2D embedding array."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        # n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


### CLUSTER SELECTION HELPERS

def silhouette_sweep(X, k_range=range(2, 9)):
    """
    Compute silhouette score for each k in k_range using KMeans.
    Returns DataFrame with k and silhouette_score.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        score  = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": score})
    return pd.DataFrame(rows)


def assign_kmeans(X, k):
    """Fit KMeans with k clusters. Returns label array."""
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    return km.fit_predict(X)


def assign_hierarchical(X, k, method="ward"):
    """Fit hierarchical clustering, cut to k clusters. Returns label array (1-indexed → 0-indexed)."""
    Z = linkage(X, method=method)
    return fcluster(Z, k, criterion="maxclust") - 1


### PLOTTING FUNCTIONS

def plot_scree(pca, n_show=15, save_path=None):
    """Scree plot: explained variance per PC + cumulative."""
    n = min(n_show, len(pca.explained_variance_ratio_))
    ev  = pca.explained_variance_ratio_[:n] * 100
    cum = np.cumsum(ev)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.bar(range(1, n+1), ev,  color=IMPACT_SIX[0], alpha=0.75, label="Per-PC variance")
    ax2.plot(range(1, n+1), cum, color=IMPACT_SIX[3], linewidth=2.5,
             marker="o", markersize=5, label="Cumulative")
    ax2.axhline(80, linestyle="--", color="#888888", linewidth=1, alpha=0.6)

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color=IMPACT_SIX[0])
    ax2.set_ylabel("Cumulative Variance (%)", color=IMPACT_SIX[3])
    ax1.set_xticks(range(1, n+1))
    ax1.set_title("Scree Plot — PCA on Item Responses", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_pca_biplot(X_pca, labels, feature_names, pc_x=0, pc_y=1,
                   color_col=None, color_map=None, pca=None,
                   n_loadings=8, save_path=None):
    """
    PCA scatter (models) with optional loading arrows for top items.
    color_col: array-like of group labels per model for coloring.
    color_map: dict {label: hex_color}.
    pca: fitted PCA object (required if n_loadings > 0).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    if color_col is not None and color_map is not None:
        unique = sorted(set(color_col))
        for i, grp in enumerate(unique):
            idx = [j for j, g in enumerate(color_col) if g == grp]
            ax.scatter(
                X_pca[idx, pc_x], X_pca[idx, pc_y],
                color=color_map.get(grp, "#aaaaaa"),
                alpha=0.75, s=55, label=str(grp), zorder=3,
            )
        ax.legend(title="Group", fontsize=11, title_fontsize=11,
                  loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        ax.scatter(X_pca[:, pc_x], X_pca[:, pc_y],
                   color=IMPACT_SIX[0], alpha=0.65, s=55, zorder=3)

    ### Loading arrows
    if pca is not None and n_loadings > 0:
        loadings = pca.components_[[pc_x, pc_y]].T
        importance = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
        top_idx = np.argsort(importance)[-n_loadings:]
        scale = np.abs(X_pca[:, [pc_x, pc_y]]).max() * 0.55
        for idx in top_idx:
            ax.annotate(
                "", xy=(loadings[idx, 0]*scale, loadings[idx, 1]*scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
            )
            ax.text(
                loadings[idx, 0]*scale*1.12,
                loadings[idx, 1]*scale*1.12,
                feature_names[idx],
                fontsize=9, ha="center", color="#333333",
            )

    ev = pca.explained_variance_ratio_ if pca is not None else [0, 0]
    ax.set_xlabel(f"PC{pc_x+1} ({ev[pc_x]*100:.1f}% var)", fontsize=13)
    ax.set_ylabel(f"PC{pc_y+1} ({ev[pc_y]*100:.1f}% var)", fontsize=13)
    ax.set_title("PCA — Model Responses (Item Space)", fontweight="bold")
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.grid(linestyle="--", alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_tsne(embedding, model_labels, color_col=None, color_map=None,
              cluster_labels=None, title="t-SNE — Model Responses",
              annotate=False, save_path=None):
    """
    t-SNE scatter. Color by color_col (group array) or cluster_labels (int array).
    annotate=True adds model name text (readable only for small N).
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    if cluster_labels is not None:
        unique_k = sorted(set(cluster_labels))
        for k in unique_k:
            idx = [i for i, c in enumerate(cluster_labels) if c == k]
            ax.scatter(
                embedding[idx, 0], embedding[idx, 1],
                color=CLUSTER_PAL[k % len(CLUSTER_PAL)],
                alpha=0.78, s=60, label=f"Cluster {k+1}", zorder=3,
            )
        ax.legend(title="Cluster", fontsize=11, title_fontsize=11)

    elif color_col is not None and color_map is not None:
        unique = sorted(set(color_col))
        for grp in unique:
            idx = [i for i, g in enumerate(color_col) if g == grp]
            ax.scatter(
                embedding[idx, 0], embedding[idx, 1],
                color=color_map.get(grp, "#aaaaaa"),
                alpha=0.78, s=60, label=str(grp), zorder=3,
            )
        ax.legend(title="Group", fontsize=11, title_fontsize=11,
                  loc="upper left", bbox_to_anchor=(1.02, 1))

    else:
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   color=IMPACT_SIX[0], alpha=0.65, s=60, zorder=3)

    # if annotate:
    #     for i, name in enumerate(model_labels):
    #         short = name.split("/")[-1]
    #         ax.text(embedding[i, 0]+0.5, embedding[i, 1]+0.5,
    #                 short, fontsize=6, alpha=0.7)

    if annotate:
        for i, name in enumerate(model_labels):
            short = name.split("/")[-1]
            ax.text(embedding[i, 0] + 0.2, embedding[i, 1] + 0.2,  # closer
                    short, fontsize=10, alpha=0.9)  # bigger

    ax.set_xlabel("t-SNE 1", fontsize=13)
    ax.set_ylabel("t-SNE 2", fontsize=13)
    ax.set_title(title, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_silhouette_sweep(df_sil, save_path=None):
    """Bar chart of silhouette scores across k values."""
    best_k = df_sil.loc[df_sil["silhouette"].idxmax(), "k"]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [IMPACT_SIX[3] if k == best_k else IMPACT_SIX[0] for k in df_sil["k"]]
    ax.bar(df_sil["k"], df_sil["silhouette"], color=colors, alpha=0.85, width=0.6)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(df_sil["k"])
    ax.set_title("Cluster Selection — Silhouette Scores", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.text(
        best_k, df_sil.loc[df_sil["k"] == best_k, "silhouette"].values[0] + 0.003,
        f"best k={best_k}", ha="center", fontsize=11, color=IMPACT_SIX[3], fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_dendrogram(X, model_labels, k_cut=None, method="ward",
                   max_display=50, save_path=None):
    """
    Hierarchical clustering dendrogram.
    k_cut: if set, draws a horizontal cut line and colors branches by cluster.
    max_display: truncate to last N merges for readability when N models is large.
    """
    Z = linkage(X, method=method)

    fig, ax = plt.subplots(figsize=(14, 5))

    dend = dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp" if len(model_labels) > max_display else None,
        p=max_display,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0,
        above_threshold_color="#888888",
        labels=model_labels if len(model_labels) <= max_display else None,
    )

    if k_cut is not None:
        # Find threshold that gives exactly k_cut clusters
        from scipy.cluster.hierarchy import fcluster
        distances = Z[:, 2]
        for thresh in sorted(distances, reverse=True):
            if len(set(fcluster(Z, thresh, criterion="distance"))) >= k_cut:
                cut_thresh = thresh
                break
        ax.axhline(cut_thresh, color=IMPACT_SIX[3], linewidth=1.8,
                   linestyle="--", label=f"k={k_cut} cut")
        ax.legend(fontsize=11)

    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward linkage)", fontweight="bold")
    ax.set_xlabel("Model" if len(model_labels) <= max_display else "Cluster node")
    ax.set_ylabel("Distance")
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_cluster_trait_profiles(df_B, cluster_labels, model_labels,
                                cols=OCEAN_COLS, save_path=None):
    """
    Mean OCEAN trait profile per cluster as a grouped bar chart.
    Also shows individual model scores as background jitter.
    """
    df_plot = df_B.copy()
    if "model" not in df_plot.columns:
        df_plot = df_plot.reset_index()

    df_plot = df_plot[df_plot["model"].isin(model_labels)].copy()
    df_plot["cluster"] = df_plot["model"].map(
        {m: c for m, c in zip(model_labels, cluster_labels)}
    )
    df_plot = df_plot.dropna(subset=["cluster"] + cols)

    k = df_plot["cluster"].nunique()
    cluster_ids = sorted(df_plot["cluster"].unique())

    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5),
                             sharey=True)

    for i, col in enumerate(cols):
        ax = axes[i]

        ### Background jitter per cluster
        for j, cid in enumerate(cluster_ids):
            sub = df_plot[df_plot["cluster"] == cid][col].dropna()
            x_jitter = np.random.normal(j, 0.08, size=len(sub))
            ax.scatter(x_jitter, sub,
                       color=CLUSTER_PAL[j % len(CLUSTER_PAL)],
                       alpha=0.25, s=20, zorder=2)

        ### Mean bar
        means = [df_plot[df_plot["cluster"] == cid][col].mean() for cid in cluster_ids]
        ax.bar(range(k), means,
               color=[CLUSTER_PAL[j % len(CLUSTER_PAL)] for j in range(k)],
               alpha=0.7, width=0.5, zorder=3)

        ax.set_ylim(1, 5)
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"C{int(c)+1}" for c in cluster_ids], fontsize=12)
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Mean Score" if i == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("OCEAN Trait Profiles by Cluster", fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_cluster_metadata_breakdown(df_metadata, cluster_labels, model_labels,
                                    meta_cols=None, save_path=None):
    """
    Stacked bar charts showing composition of each cluster by metadata variables
    (e.g. license_group, Reasoning, Country / Region).
    """
    if meta_cols is None:
        meta_cols = ["license_group", "Reasoning"]

    ### Build cluster assignment df
    df_clust = pd.DataFrame({
        "model": model_labels,
        "cluster": [f"C{int(c)+1}" for c in cluster_labels],
    })

    # Identify model column in metadata
    model_col = "model" if "model" in df_metadata.columns else "Model_ID"
    df_meta = df_metadata[[model_col] + meta_cols].rename(columns={model_col: "model"})
    df_clust = df_clust.merge(df_meta, on="model", how="left")

    n_cols  = len(meta_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(meta_cols):
        ax = axes[i]
        ct = (
            df_clust.groupby(["cluster", col])
            .size()
            .unstack(col)
            .fillna(0)
        )
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

        ct_pct.plot(
            kind="bar", stacked=True, ax=ax,
            color=CLUSTER_PAL[:len(ct_pct.columns)],
            alpha=0.85, width=0.6, legend=True,
        )
        ax.set_title(f"Cluster composition — {col}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% of cluster" if i == 0 else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title=col, fontsize=10, title_fontsize=10,
                  loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig
