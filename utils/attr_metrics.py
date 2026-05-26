"""
Attribute-aware evaluation metrics for Referring Expression Counting.

Provides three novel metrics:
  - AD-MAE  (Attribute Discrimination MAE)
  - ACR     (Attribute Confusion Rate)
  - APDR    (Attribute Pair Discrimination Rate)

Plus a helper that computes a full structured report from DataFrames.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_ad_mae(pred_a, pred_b, gt_a, gt_b):
    """Attribute Discrimination MAE.

    Measures how well the model distinguishes the *difference* in count
    between two attribute-variant queries on the same image.

    .. math::
        \\text{AD-MAE} = \\frac{1}{N} \\sum_{i=1}^{N}
            \\bigl| |\\hat{c}_a^{(i)} - \\hat{c}_b^{(i)}|
                   - |c_a^{(i)} - c_b^{(i)}| \\bigr|

    Args:
        pred_a: array-like of predicted counts for attribute a.
        pred_b: array-like of predicted counts for attribute b.
        gt_a:   array-like of ground-truth counts for attribute a.
        gt_b:   array-like of ground-truth counts for attribute b.

    Returns:
        float: AD-MAE value.  Returns 0.0 when inputs are empty.
    """
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    gt_a = np.asarray(gt_a, dtype=np.float64)
    gt_b = np.asarray(gt_b, dtype=np.float64)

    if pred_a.size == 0:
        return 0.0

    pred_diff = np.abs(pred_a - pred_b)
    gt_diff = np.abs(gt_a - gt_b)
    return float(np.mean(np.abs(pred_diff - gt_diff)))


def compute_acr(pred_a, pred_b, gt_a, gt_b, threshold=3, epsilon=2):
    """Attribute Confusion Rate.

    Among image pairs where the ground-truth counts differ by more than
    *threshold*, ACR is the fraction for which the model's predicted
    difference is at most *epsilon* (i.e. the model fails to tell the
    two attributes apart).

    .. math::
        \\text{ACR} = \\frac
            {\\#\\{i : |c_a-c_b| > T \\;\\wedge\\; |\\hat c_a-\\hat c_b| \\le \\varepsilon\\}}
            {\\#\\{i : |c_a-c_b| > T\\}}

    Args:
        pred_a:    array-like of predicted counts for attribute a.
        pred_b:    array-like of predicted counts for attribute b.
        gt_a:      array-like of ground-truth counts for attribute a.
        gt_b:      array-like of ground-truth counts for attribute b.
        threshold: minimum GT difference to consider a pair "eligible".
        epsilon:   maximum predicted difference to count as "confused".

    Returns:
        float: ACR value in [0, 1].  Returns 0.0 when no eligible pairs.
    """
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    gt_a = np.asarray(gt_a, dtype=np.float64)
    gt_b = np.asarray(gt_b, dtype=np.float64)

    gt_diff = np.abs(gt_a - gt_b)
    eligible = gt_diff > threshold

    if eligible.sum() == 0:
        return 0.0

    pred_diff = np.abs(pred_a - pred_b)
    confused = pred_diff[eligible] <= epsilon
    return float(confused.sum() / eligible.sum())


def compute_apdr(pred_a, pred_b, gt_a, gt_b):
    """Attribute Pair Discrimination Rate.

    For each pair, check whether the model correctly predicts which
    attribute has a higher count (or ties).  When the ground-truth
    counts are equal, any prediction direction is accepted.

    .. math::
        \\text{APDR} = \\frac{1}{N} \\sum_{i=1}^{N}
            \\mathbb{1}\\bigl[
                \\operatorname{sign}(\\hat c_a - \\hat c_b)
                = \\operatorname{sign}(c_a - c_b)
                \\;\\vee\\; c_a = c_b
            \\bigr]

    Args:
        pred_a: array-like of predicted counts for attribute a.
        pred_b: array-like of predicted counts for attribute b.
        gt_a:   array-like of ground-truth counts for attribute a.
        gt_b:   array-like of ground-truth counts for attribute b.

    Returns:
        float: APDR value in [0, 1].  Returns 0.0 when inputs are empty.
    """
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    gt_a = np.asarray(gt_a, dtype=np.float64)
    gt_b = np.asarray(gt_b, dtype=np.float64)

    if pred_a.size == 0:
        return 0.0

    gt_dir = np.sign(gt_a - gt_b)
    pred_dir = np.sign(pred_a - pred_b)

    tie = gt_a == gt_b
    correct = (gt_dir == pred_dir) | tie

    return float(correct.mean())


# ---------------------------------------------------------------------------
# Helpers for grouped computation
# ---------------------------------------------------------------------------

def _entry_metrics(df):
    """Compute entry-level metrics (MAE, RMSE) from a slice of entries_df."""
    n = len(df)
    if n == 0:
        return {"num_entries": 0, "mae": 0.0, "rmse": 0.0}
    ae = np.asarray(df["ae"], dtype=np.float64)
    return {
        "num_entries": n,
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((ae ** 2).mean())),
    }


def _pair_metrics(df, threshold=3, epsilon=2):
    """Compute pair-level metrics (AD-MAE, ACR, APDR) from a slice of pairs_df."""
    n = len(df)
    if n == 0:
        return {"num_pairs": 0, "ad_mae": 0.0, "acr": 0.0, "apdr": 0.0}
    pa = df["pred_count_a"].values
    pb = df["pred_count_b"].values
    ga = df["gt_count_a"].values
    gb = df["gt_count_b"].values
    return {
        "num_pairs": n,
        "ad_mae": compute_ad_mae(pa, pb, ga, gb),
        "acr": compute_acr(pa, pb, ga, gb, threshold=threshold, epsilon=epsilon),
        "apdr": compute_apdr(pa, pb, ga, gb),
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def compute_all_attr_metrics(entries_df, pairs_df, threshold=3, epsilon=2):
    """Compute a structured metrics report from prediction DataFrames.

    Args:
        entries_df: :class:`pandas.DataFrame` with columns
            ``image_id``, ``type``, ``gt_count``, ``pred_count``, ``ae``.
        pairs_df: :class:`pandas.DataFrame` with columns
            ``image_id``, ``type``, ``difficulty_level``,
            ``gt_count_a``, ``gt_count_b``, ``pred_count_a``, ``pred_count_b``.
        threshold: ACR threshold (default 3).
        epsilon:   ACR epsilon  (default 2).

    Returns:
        dict: Nested report with keys ``overall``, ``by_type``,
        ``by_difficulty``, and ``by_type_difficulty``.
    """
    report = {}

    # --- overall -----------------------------------------------------------
    overall_entry = _entry_metrics(entries_df)
    overall_pair = _pair_metrics(pairs_df, threshold, epsilon)
    report["overall"] = {
        **overall_entry,
        "num_pairs": overall_pair["num_pairs"],
        "ad_mae": overall_pair["ad_mae"],
        "acr": overall_pair["acr"],
        "apdr": overall_pair["apdr"],
    }

    # --- by_type -----------------------------------------------------------
    by_type = {}
    if len(entries_df) > 0:
        for t, grp_e in entries_df.groupby("type"):
            grp_p = pairs_df[pairs_df["type"] == t] if len(pairs_df) > 0 else pairs_df
            em = _entry_metrics(grp_e)
            pm = _pair_metrics(grp_p, threshold, epsilon)
            by_type[t] = {
                "num_entries": em["num_entries"],
                "mae": em["mae"],
                "ad_mae": pm["ad_mae"],
                "acr": pm["acr"],
                "apdr": pm["apdr"],
            }
    report["by_type"] = by_type

    # --- by_difficulty -----------------------------------------------------
    by_diff = {}
    if len(pairs_df) > 0:
        for d, grp in pairs_df.groupby("difficulty_level"):
            by_diff[d] = _pair_metrics(grp, threshold, epsilon)
    report["by_difficulty"] = by_diff

    # --- by_type_difficulty ------------------------------------------------
    by_td = {}
    if len(pairs_df) > 0:
        for t, grp_t in pairs_df.groupby("type"):
            by_td[t] = {}
            for d, grp_d in grp_t.groupby("difficulty_level"):
                by_td[t][d] = _pair_metrics(grp_d, threshold, epsilon)
    report["by_type_difficulty"] = by_td

    return report
