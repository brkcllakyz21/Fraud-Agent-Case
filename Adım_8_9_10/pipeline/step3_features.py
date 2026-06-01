from __future__ import annotations

"""
Adım 3 — Feature Engineering
Notebook'tan .py'ye dönüştürülmüş, inference uyumlu hale getirilmiş.
Tek satır inference için target_col=None geçilir.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def detect_column_types(df, unique_threshold=20, unique_ratio_threshold=0.01):
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n if n > 0 else 0
        categorical_patterns = ("card", "addr", "ProductCD", "id_",
                                 "DeviceType", "DeviceInfo", "P_emaildomain",
                                 "R_emaildomain", "M")
        is_name_categorical = col.startswith(categorical_patterns)
        if is_name_categorical:
            semantic_type = "categorical (name-override)"
        elif pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_float_dtype(series) and n_unique > unique_threshold:
                semantic_type = "numeric"
            elif n_unique <= unique_threshold or unique_ratio < unique_ratio_threshold:
                semantic_type = "categorical (numeric-coded)"
            else:
                semantic_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            semantic_type = "datetime"
        else:
            semantic_type = "categorical"
        rows.append({
            "column": col,
            "pandas_dtype": str(dtype),
            "n_unique": n_unique,
            "unique_ratio (%)": round(unique_ratio * 100, 2),
            "semantic_type": semantic_type,
        })
    return pd.DataFrame(rows)


def create_temporal_features(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    if time_col not in df.columns:
        return df
    out = df.copy()
    seconds = out[time_col]
    out["tx_hour"]          = (seconds // 3600) % 24
    out["tx_day_of_week"]   = (seconds // 86400) % 7
    out["tx_is_weekend"]    = out["tx_day_of_week"].isin([5, 6]).astype(int)
    out["tx_is_night"]      = out["tx_hour"].between(0, 5).astype(int)
    out["tx_is_business"]   = (
        (out["tx_hour"].between(9, 17)) & (out["tx_is_weekend"] == 0)
    ).astype(int)

    def _day_part(hour):
        if 6 <= hour < 12:   return "morning"
        elif 12 <= hour < 18: return "afternoon"
        elif 18 <= hour < 24: return "evening"
        else:                 return "night"

    out["tx_day_part"]          = out["tx_hour"].apply(_day_part)
    out["tx_days_since_start"]  = (seconds - seconds.min()) // 86400
    return out


def create_entity_features(
    df: pd.DataFrame,
    entity_cols: Optional[list] = None,
    amount_col: str = "TransactionAmt",
    time_col: str = "TransactionDT",
    min_tx_count: int = 3,
) -> pd.DataFrame:
    if entity_cols is None:
        entity_cols = ["card1", "P_emaildomain", "DeviceType"]
    out = df.copy()
    global_amt_mean = out[amount_col].mean()
    global_amt_std  = out[amount_col].std()

    for entity in entity_cols:
        if entity not in out.columns:
            continue
        prefix = entity.lower().replace("_", "")
        stats = (
            out.groupby(entity)[amount_col]
            .agg(tx_count="count", amt_mean="mean", amt_std="std", amt_median="median")
            .reset_index()
        )
        stats.columns = [
            entity,
            f"{prefix}_tx_count", f"{prefix}_amt_mean",
            f"{prefix}_amt_std",  f"{prefix}_amt_median",
        ]
        stats[f"{prefix}_amt_std"] = stats[f"{prefix}_amt_std"].fillna(global_amt_std)
        out = out.merge(stats, on=entity, how="left")
        out[f"{prefix}_amt_zscore"] = (
            (out[amount_col] - out[f"{prefix}_amt_mean"])
            / out[f"{prefix}_amt_std"].replace(0, global_amt_std)
        )
        if time_col in out.columns:
            out = out.sort_values([entity, time_col])
            for window, label in [(3600, "1h"), (86400, "1d")]:
                col_name = f"{prefix}_velocity_{label}"
                out[col_name] = (
                    out.groupby(entity)[time_col]
                    .transform(
                        lambda t: t.expanding().apply(
                            lambda x: ((x[-1] - x) <= window).sum(), raw=True
                        )
                    )
                )
    return out


def create_relational_features(
    df: pd.DataFrame,
    amount_col: str = "TransactionAmt",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    out = df.copy()
    if "C1" in out.columns:
        out["rel_amt_per_c1"] = out[amount_col] / (out["C1"].replace(0, np.nan))
    if all(c in out.columns for c in ["C1", "C2"]):
        out["rel_c1_c2_ratio"] = out["C1"] / (out["C2"].replace(0, np.nan))
    if all(c in out.columns for c in ["D1", "D2"]):
        out["rel_d1_d2_diff"] = out["D1"] - out["D2"]
    if all(c in out.columns for c in ["D1", "D3"]):
        out["rel_d1_d3_diff"] = out["D1"] - out["D3"]
    if "D1" in out.columns and time_col in out.columns:
        days_elapsed = (out[time_col] - out[time_col].min()) / 86400
        out["rel_d1_age_ratio"] = out["D1"] / (days_elapsed.replace(0, np.nan))
    if all(c in out.columns for c in ["card1", "addr1"]):
        combo = out.groupby(["card1", "addr1"]).size().reset_index(name="rel_card_addr_count")
        out = out.merge(combo, on=["card1", "addr1"], how="left")
        addr_diversity = (
            out.groupby("card1")["addr1"].nunique()
            .reset_index(name="rel_card_addr_diversity")
        )
        out = out.merge(addr_diversity, on="card1", how="left")
    if all(c in out.columns for c in ["P_emaildomain", "R_emaildomain"]):
        out["rel_email_match"] = (out["P_emaildomain"] == out["R_emaildomain"]).astype(int)
        out["rel_email_both_known"] = (
            out["P_emaildomain"].notna() & out["R_emaildomain"].notna()
        ).astype(int)
    if "tx_is_night" in out.columns:
        out["rel_night_high_amt"] = out["tx_is_night"] * out[amount_col]
    if "tx_is_business" in out.columns:
        out["rel_business_amt"] = out["tx_is_business"] * out[amount_col]
    return out


def create_context_features(
    df: pd.DataFrame,
    amount_col: str = "TransactionAmt",
    target_col: Optional[str] = "isFraud",
    rare_threshold: float = 0.01,
    high_lift_threshold: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()
    global_fraud_rate = out[target_col].mean() if target_col and target_col in out.columns else None
    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]

    for col in cat_cols:
        if col not in out.columns:
            continue
        freq = out[col].value_counts(normalize=True)
        rare_vals = freq[freq < rare_threshold].index
        feat_name = f"ctx_{col.lower().replace('_', '')}_is_rare"
        out[feat_name] = out[col].isin(rare_vals).astype(int)
        freq_feat = f"ctx_{col.lower().replace('_', '')}_freq"
        out[freq_feat] = out[col].map(freq)
        if target_col and target_col in out.columns and global_fraud_rate:
            fraud_rate_map = out.groupby(col)[target_col].mean()
            lift_map = fraud_rate_map / global_fraud_rate
            lift_feat = f"ctx_{col.lower().replace('_', '')}_fraud_lift"
            out[lift_feat] = out[col].map(lift_map)

    if "ProductCD" in out.columns:
        prod_amt_mean = out.groupby("ProductCD")[amount_col].mean()
        out["ctx_product_amt_mean"]  = out["ProductCD"].map(prod_amt_mean)
        out["ctx_product_amt_diff"]  = out[amount_col] - out["ctx_product_amt_mean"]
        out["ctx_product_amt_ratio"] = out[amount_col] / out["ctx_product_amt_mean"].replace(0, np.nan)

    if "card4" in out.columns:
        card4_amt_mean = out.groupby("card4")[amount_col].mean()
        out["ctx_card4_amt_mean"] = out["card4"].map(card4_amt_mean)
        out["ctx_card4_amt_diff"] = out[amount_col] - out["ctx_card4_amt_mean"]

    if all(c in out.columns for c in ["ProductCD", "card4"]):
        combo_key = out["ProductCD"].astype(str) + "_" + out["card4"].astype(str)
        combo_freq = combo_key.value_counts(normalize=True)
        out["ctx_prod_card4_combo_freq"] = combo_key.map(combo_freq)
        out["ctx_prod_card4_is_rare"]    = (out["ctx_prod_card4_combo_freq"] < rare_threshold).astype(int)
        if target_col and target_col in out.columns and global_fraud_rate:
            combo_fraud = out.groupby(combo_key)[target_col].mean()
            combo_lift  = combo_fraud / global_fraud_rate
            out["ctx_prod_card4_lift"]      = combo_key.map(combo_lift)
            out["ctx_prod_card4_high_risk"] = (
                (out["ctx_prod_card4_is_rare"] == 1) &
                (out["ctx_prod_card4_lift"] >= high_lift_threshold)
            ).astype(int)

    global_mean = out[amount_col].mean()
    global_std  = out[amount_col].std()
    out["ctx_amt_global_zscore"] = (out[amount_col] - global_mean) / global_std
    out["ctx_amt_log"]           = np.log1p(out[amount_col])
    return out


def run_feature_engineering(
    df: pd.DataFrame,
    entity_cols: Optional[list] = None,
    amount_col: str = "TransactionAmt",
    time_col: str = "TransactionDT",
    target_col: Optional[str] = None,
    rare_threshold: float = 0.01,
    high_lift_threshold: float = 2.0,
    min_tx_count: int = 3,
) -> pd.DataFrame:
    df = create_temporal_features(df, time_col=time_col)
    df = create_entity_features(df, entity_cols=entity_cols, amount_col=amount_col,
                                 time_col=time_col, min_tx_count=min_tx_count)
    df = create_relational_features(df, amount_col=amount_col, time_col=time_col)
    df = create_context_features(df, amount_col=amount_col, target_col=target_col,
                                  rare_threshold=rare_threshold,
                                  high_lift_threshold=high_lift_threshold)
    return df