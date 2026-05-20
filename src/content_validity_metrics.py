import numpy as np
import pandas as pd
import pingouin as pg
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa as statsmodels_fleiss, aggregate_raters


def mean(data: pd.DataFrame):
    """Returns mean."""
    mean = round(data.mean(), 2)
    return mean


def sd(data: pd.DataFrame):
    """Returns standard deviation."""
    sd = round(data.std(), 2)
    return sd


def i_cvi(data: pd.DataFrame, facet: str):
    """Returns I-CVI per item."""

    wide = data.pivot(index="item_id", columns="rater", values=facet)

    item_icvi = wide.apply(
        lambda row: row.isin([3, 4]).mean(),
        axis=1
    )

    return item_icvi.round(2)


def s_cvi(data: pd.DataFrame, facet: str):
    """Return S-CVI (average) = average of item-level CVIs."""

    item_icvi = i_cvi(data, facet)

    return round(item_icvi.mean(), 2)


def gwet_ac2(data: pd.DataFrame, facet: str):
    """Returns Gwet's AC2."""
    wide = data.pivot(index="item_id", columns="rater", values=facet).dropna()
    vals = wide.values.astype(float)
    cats = sorted(np.unique(vals[~np.isnan(vals)]))
    K = len(cats)

    if K <= 1:
        return np.nan

    idx = {c: i for i, c in enumerate(cats)}
    n, r = vals.shape

    # n_mat[i, k] = number of raters assigning category k to item i
    n_mat = np.zeros((n, K))
    for i in range(n):
        for j in range(r):
            n_mat[i, idx[vals[i, j]]] += 1

    # Linear weight matrix
    W = np.array([[1 - abs(a - b) / (K - 1) for b in range(K)] for a in range(K)])

    # Weighted observed agreement (over all ordered rater pairs)
    p_o = 0.0
    for i in range(n):
        s = sum(W[k, l] * n_mat[i, k] * (n_mat[i, l] - (1 if k == l else 0))
                for k in range(K) for l in range(K))
        p_o += s / (r * (r - 1))
    p_o /= n

    # Gwet's chance agreement (2008, eq. 5)
    pi = n_mat.sum(axis=0) / n_mat.sum()
    p_e = sum(W[k, l] * pi[k] * (1 - pi[l]) for k in range(K) for l in range(K)) / K

    return round((p_o - p_e) / (1 - p_e), 2)

def list_to_df(lst, col):
    """Returns df from list."""
    return pd.DataFrame(lst, columns=["questionnaire", col]).set_index("questionnaire")
