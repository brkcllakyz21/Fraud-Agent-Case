from __future__ import annotations

"""
Adım 5 — Score Aggregation
Notebook'tan .py'ye dönüştürülmüş.
target_col=None → default ağırlıklar kullanılır (inference modu).
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DEFAULT_WEIGHTS = {
    "multivariate_score":  0.30,
    "entity_score":        0.25,
    "column_score_max":    0.20,
    "temporal_score":      0.15,
    "column_score_mean":   0.10,
}


def robust_normalize(series: pd.Series) -> pd.Series:
    series = series.fillna(0.0)
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    if iqr == 0:
        return pd.Series(0.0, index=series.index)
    normalized = (series - series.median()) / iqr
    return normalized.clip(0, 1)


def compute_auc_weights(
    df: pd.DataFrame,
    score_cols: List[str],
    target_col: str,
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    aucs = {}
    y_true = df[target_col]
    for col in score_cols:
        try:
            auc = roc_auc_score(y_true, df[col])
            aucs[col] = max(0.0, auc - 0.5)
        except Exception:
            aucs[col] = 0.0
    total = sum(aucs.values())
    if total == 0:
        return {}
    return {col: val / total for col, val in aucs.items()}


def normalize_scores(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    df_norm = df.copy()
    for col in score_cols:
        if col in df_norm.columns:
            df_norm[col] = robust_normalize(df_norm[col])
        else:
            df_norm[col] = 0.0
    return df_norm


def aggregate_scores(
    df: pd.DataFrame,
    score_cols: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    target_col: Optional[str] = None,
    output_col: str = "fraud_score",
) -> Tuple[pd.DataFrame, dict]:
    if score_cols is None:
        score_cols = list(DEFAULT_WEIGHTS.keys())

    df_norm = normalize_scores(df, score_cols)
    mode = "default"
    auc_scores_report = None

    if weights is not None:
        total = sum(weights.values())
        final_weights = {col: w / total for col, w in weights.items() if col in score_cols}
        mode = "manual"
    elif target_col is not None and target_col in df.columns:
        auc_weights = compute_auc_weights(df_norm, score_cols, target_col)
        if auc_weights:
            final_weights = auc_weights
            mode = "auc"
            from sklearn.metrics import roc_auc_score
            auc_scores_report = {}
            y_true = df[target_col]
            for col in score_cols:
                try:
                    auc_scores_report[col] = round(roc_auc_score(y_true, df_norm[col]), 4)
                except Exception:
                    auc_scores_report[col] = None
        else:
            final_weights = {col: DEFAULT_WEIGHTS.get(col, 0.0) for col in score_cols}
    else:
        final_weights = {col: DEFAULT_WEIGHTS.get(col, 0.0) for col in score_cols}

    for col in score_cols:
        if col not in final_weights:
            final_weights[col] = 0.0

    total_w = sum(final_weights.values())
    if total_w == 0:
        raise ValueError("Tüm ağırlıklar sıfır.")
    final_weights = {col: w / total_w for col, w in final_weights.items()}

    df_out = df.copy()
    df_out[output_col] = sum(
        df_norm[col] * final_weights.get(col, 0.0)
        for col in score_cols
        if col in df_norm.columns
    ).clip(0, 1)

    contribution = {
        col: round(
            (df_norm[col] * final_weights.get(col, 0.0)).mean() / df_out[output_col].mean()
            if df_out[output_col].mean() > 0 else 0.0, 4
        )
        for col in score_cols if col in df_norm.columns
    }

    weight_report = {
        "mode": mode,
        "weights": {col: round(w, 4) for col, w in final_weights.items()},
        "auc_scores": auc_scores_report,
        "contribution": contribution,
    }
    return df_out, weight_report