from __future__ import annotations

"""
Adım 6 — Context Adjust Engine
Notebook'tan .py'ye dönüştürülmüş.
target_col=None → separation metrikleri hesaplanmaz (inference modu).
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCORE_COL  = "fraud_score"
TARGET_COL = "isFraud"


def apply_business_hours_context(df: pd.DataFrame, tx_hour_col: str = "tx_hour") -> pd.Series:
    hour = df[tx_hour_col]
    multiplier = pd.Series(1.00, index=df.index)
    multiplier = multiplier.where(~((hour >= 8) & (hour < 18)), other=0.85)
    multiplier = multiplier.where(~((hour >= 22) | (hour < 8)), other=1.15)
    return multiplier


def apply_weekend_context(
    df: pd.DataFrame,
    weekend_col: str = "tx_is_weekend",
    amt_col: str = "TransactionAmt",
    amt_threshold: Optional[float] = None,
) -> pd.Series:
    if amt_threshold is None:
        amt_threshold = df[amt_col].median()
    is_weekend  = df[weekend_col] == 1
    is_high_amt = df[amt_col] > amt_threshold
    multiplier = pd.Series(1.00, index=df.index)
    multiplier = multiplier.where(~(is_weekend & ~is_high_amt), other=0.95)
    multiplier = multiplier.where(~(is_weekend &  is_high_amt), other=1.10)
    return multiplier


def apply_trusted_entity_context(
    df: pd.DataFrame,
    velocity_col: str   = "card1_velocity_1d",
    amt_zscore_col: str = "card1_amt_zscore",
    max_velocity: float = 3.0,
    max_zscore: float   = 0.3,
) -> pd.Series:
    multiplier = pd.Series(1.00, index=df.index)
    if velocity_col not in df.columns or amt_zscore_col not in df.columns:
        return multiplier
    is_low_velocity = df[velocity_col] <= max_velocity
    is_stable_amt   = df[amt_zscore_col].abs() <= max_zscore
    multiplier = multiplier.where(~(is_low_velocity & is_stable_amt), other=0.85)
    return multiplier


def apply_transaction_type_context(
    df: pd.DataFrame,
    fraud_lift_col: str = "ctx_productcd_fraud_lift",
    lift_low: float     = 0.8,
    lift_high: float    = 1.2,
    mult_low: float     = 0.90,
    mult_high: float    = 1.20,
) -> pd.Series:
    multiplier = pd.Series(1.00, index=df.index)
    if fraud_lift_col not in df.columns:
        return multiplier
    lift = df[fraud_lift_col]
    multiplier = multiplier.where(~(lift < lift_low),  other=mult_low)
    multiplier = multiplier.where(~(lift > lift_high), other=mult_high)
    return multiplier


def apply_high_value_night_context(
    df: pd.DataFrame,
    tx_hour_col: str      = "tx_hour",
    amt_col: str          = "TransactionAmt",
    night_start: int      = 22,
    night_end: int        = 6,
    amt_percentile: float = 95.0,
    multiplier_val: float = 1.30,
) -> pd.Series:
    amt_threshold = np.percentile(df[amt_col].dropna(), amt_percentile)
    hour     = df[tx_hour_col]
    is_night = (hour >= night_start) | (hour < night_end)
    is_high  = df[amt_col] > amt_threshold
    multiplier = pd.Series(1.00, index=df.index)
    multiplier = multiplier.where(~(is_night & is_high), other=multiplier_val)
    return multiplier


def run_context_adjustment(
    df: pd.DataFrame,
    score_col: str        = SCORE_COL,
    target_col: Optional[str] = None,
    tx_hour_col: str      = "tx_hour",
    weekend_col: str      = "tx_is_weekend",
    amt_col: str          = "TransactionAmt",
    velocity_col: str     = "card1_velocity_1d",
    amt_zscore_col: str   = "card1_amt_zscore",
    fraud_lift_col: str   = "ctx_productcd_fraud_lift",
    max_velocity: float   = 6.0,
    max_zscore: float     = 0.5,
    amt_percentile: float = 95.0,
    lift_low: float       = 0.80,
    lift_high: float      = 1.20,
) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()

    df["business_hours_multiplier"]   = apply_business_hours_context(df, tx_hour_col=tx_hour_col)
    df["weekend_multiplier"]          = apply_weekend_context(df, weekend_col=weekend_col, amt_col=amt_col)
    df["trusted_entity_multiplier"]   = apply_trusted_entity_context(
        df, velocity_col=velocity_col, amt_zscore_col=amt_zscore_col,
        max_velocity=max_velocity, max_zscore=max_zscore,
    )
    df["product_type_multiplier"]     = apply_transaction_type_context(
        df, fraud_lift_col=fraud_lift_col, lift_low=lift_low, lift_high=lift_high,
    )
    df["high_value_night_multiplier"] = apply_high_value_night_context(
        df, tx_hour_col=tx_hour_col, amt_col=amt_col, amt_percentile=amt_percentile,
    )

    multiplier_cols = [
        "business_hours_multiplier", "weekend_multiplier",
        "trusted_entity_multiplier", "product_type_multiplier",
        "high_value_night_multiplier",
    ]
    df["composite_multiplier"]    = df[multiplier_cols].prod(axis=1)
    df["context_adjusted_score"]  = (df[score_col] * df["composite_multiplier"]).clip(0, 1)

    total = len(df)
    report: dict = {"total_rows": total, "layers": {}}
    for col in multiplier_cols:
        affected = int((df[col] != 1.00).sum())
        report["layers"][col] = {
            "affected_rows": affected,
            "affected_pct":  round(affected / total * 100, 2),
            "pushed_up":     int((df[col] > 1.00).sum()),
            "pushed_down":   int((df[col] < 1.00).sum()),
            "mean_multiplier": round(float(df[col].mean()), 4),
        }

    if target_col and target_col in df.columns:
        def sep(scores, labels):
            fm = scores[labels == 1].mean()
            nm = scores[labels == 0].mean()
            return round(float(fm / nm), 4) if nm > 0 else None
        report["separation_before"] = sep(df[score_col], df[target_col])
        report["separation_after"]  = sep(df["context_adjusted_score"], df[target_col])

    return df, report