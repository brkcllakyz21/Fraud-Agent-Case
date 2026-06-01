from __future__ import annotations

"""
Adım 4 — Multi-Layer Anomali Detection Engine
Notebook'tan .py'ye dönüştürülmüş. run_anomaly_detection() tuple döndürür: (df, explanations, fs_report)
PipelineRunner sadece df'yi kullanır.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def feature_selection_report(
    df: pd.DataFrame,
    numeric_cols: Optional[list] = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    target_col: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target_col]
    available = [c for c in numeric_cols if c in df.columns]
    work = df[available].copy()
    scaler = MinMaxScaler()
    try:
        scaled = pd.DataFrame(
            scaler.fit_transform(work.fillna(work.median())),
            columns=work.columns,
        )
    except Exception:
        scaled = work.fillna(0)
    variances = scaled.var()
    low_var_mask = variances < variance_threshold
    dropped_low_var = variances[low_var_mask].index.tolist()
    passed_var = [c for c in available if c not in dropped_low_var]
    dropped_high_corr = {}
    if len(passed_var) > 1:
        corr_matrix = scaled[passed_var].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            for partner in upper.index[upper[col] > correlation_threshold].tolist():
                if partner not in to_drop:
                    if variances.get(col, 0) < variances.get(partner, 0):
                        to_drop.add(col)
                        dropped_high_corr[col] = f"corr with {partner}"
                    else:
                        to_drop.add(partner)
                        dropped_high_corr[partner] = f"corr with {col}"
        selected_features = [c for c in passed_var if c not in to_drop]
    else:
        selected_features = passed_var
    return {
        "selected_features": selected_features,
        "dropped_low_var": dropped_low_var,
        "dropped_high_corr": dropped_high_corr,
    }


def detect_column_anomalies(
    df: pd.DataFrame,
    numeric_cols: Optional[list] = None,
    target_col: Optional[str] = None,
    z_score_cap: float = 10.0,
    top_n_drivers: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target_col]
    available = [c for c in numeric_cols if c in df.columns]
    if not available:
        df["column_score_max"] = 0.0
        df["column_score_mean"] = 0.0
        df["column_drivers"] = ""
        return df
    work = df[available].copy()
    filled = work.fillna(work.median())
    z_scores = filled.apply(lambda col: np.abs(stats.zscore(col, nan_policy="omit")), axis=0)
    z_scores = z_scores.clip(0, z_score_cap)
    z_normalized = z_scores / z_score_cap
    df["column_score_max"]  = z_normalized.max(axis=1).round(4)
    df["column_score_mean"] = z_normalized.mean(axis=1).round(4)

    def get_drivers(row):
        top = row.nlargest(top_n_drivers)
        top = top[top > 0]
        return ", ".join([f"{col}:{val:.2f}" for col, val in top.items()]) if not top.empty else ""

    df["column_drivers"] = z_normalized.apply(get_drivers, axis=1)
    return df


def detect_multivariate_anomalies(
    df: pd.DataFrame,
    selected_features: list,
    contamination: float = 0.035,
    n_estimators: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    model_dir: str = "pipeline/models",
) -> pd.DataFrame:
    """
    Isolation Forest tabanlı çok değişkenli anomali skoru.

    Train modunda (model dosyası yoksa veya güncellenmesi gerekiyorsa):
      - IsolationForest fit edilir
      - Model, scaler ve feature listesi model_dir altına kaydedilir
        (varsa güncellenir, yoksa oluşturulur)

    Inference modunda (model dosyası varsa):
      - Kayıtlı model ve scaler yüklenir
      - Train dağılımıyla tutarlı skor üretilir

    contamination=0.035 → veri setindeki fraud oranıyla hizalı.
    """
    import json as _json
    from pathlib import Path as _Path
    import joblib
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()
    available = [c for c in selected_features if c in df.columns]

    if len(available) < 2:
        print("[UYARI] Yeterli feature yok, multivariate skor üretilemiyor.")
        df["multivariate_score"] = 0.0
        return df

    out_dir       = _Path(model_dir)
    model_path    = out_dir / "isolation_forest.joblib"
    scaler_path   = out_dir / "if_scaler.joblib"
    features_path = out_dir / "if_feature_list.json"

    # ── Kayıtlı model var mı? ─────────────────────────────────────────────
    model_loaded = False
    model = scaler = None
    score_min = score_max = None
    use_features = available

    if model_path.exists() and scaler_path.exists():
        try:
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            if features_path.exists():
                with open(features_path) as f:
                    meta = _json.load(f)
                saved = meta.get("features", available)
                use_features = [c for c in saved if c in df.columns]
                score_min = meta.get("score_min")
                score_max = meta.get("score_max")
            model_loaded = True
        except Exception:
            model_loaded = False

    work = df[use_features].fillna(df[use_features].median())

    if model_loaded and scaler is not None:
        # ── Inference: kayıtlı model ile skor üret ────────────────────────
        X_scaled   = scaler.transform(work)
        raw_scores = model.score_samples(X_scaled)
        inverted   = -raw_scores
        if score_min is not None and score_max is not None and score_max > score_min:
            normalized = np.clip((inverted - score_min) / (score_max - score_min), 0, 1)
        else:
            min_s, max_s = inverted.min(), inverted.max()
            normalized = (inverted - min_s) / (max_s - min_s) if max_s > min_s else np.zeros(len(inverted))
        _mode = "inference (loaded)"
    else:
        # ── Train: fit et ve kaydet ───────────────────────────────────────
        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(work)
        model    = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                                   random_state=random_state, n_jobs=-1)
        raw_scores = model.fit(X_scaled).score_samples(X_scaled)
        inverted   = -raw_scores
        min_s, max_s = float(inverted.min()), float(inverted.max())
        normalized = (inverted - min_s) / (max_s - min_s) if max_s > min_s else np.zeros(len(inverted))
        score_min, score_max = min_s, max_s

        # ── Kaydet (varsa güncelle, yoksa oluştur) ────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(features_path, "w", encoding="utf-8") as f:
            _json.dump({
                "features": use_features,
                "n_features": len(use_features),
                "contamination": contamination,
                "score_min": score_min,
                "score_max": score_max,
            }, f, indent=2)
        _mode = "train (fit + saved)"

    df["multivariate_score"] = np.round(normalized, 4)

    if verbose:
        print("=" * 60)
        print("MULTIVARIATE ANOMALI DETECTION (Isolation Forest)")
        print("=" * 60)
        print(f"  Mod                       : {_mode}")
        print(f"  Kullanılan feature sayısı : {len(use_features)}")
        print(f"  Contamination             : {contamination}")
        print(f"  Score — mean              : {df['multivariate_score'].mean():.4f}")
        print(f"  Score — max               : {df['multivariate_score'].max():.4f}")
        high = (df["multivariate_score"] > 0.7).sum()
        print(f"  Yüksek anomali (>0.7)     : {high:,} satır ({high/len(df)*100:.2f}%)")
        if not model_loaded:
            print(f"  Model kaydedildi          : {model_path}")
            print(f"  Scaler kaydedildi         : {scaler_path}")
            print(f"  Feature listesi           : {features_path}")
        print("=" * 60)

    return df


def detect_entity_anomalies(
    df: pd.DataFrame,
    entity_cols: list,
    target_col: Optional[str] = None,
    min_tx_count: int = 5,
    velocity_col_1h: Optional[str] = None,
    velocity_col_1d: Optional[str] = None,
    amt_zscore_suffix: str = "_amt_zscore",
    velocity_cap: float = 20.0,
    top_n_drivers: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    entity_scores: dict = {}
    for entity in entity_cols:
        if entity not in df.columns:
            continue
        prefix = entity.lower().replace("_", "")
        tx_count_col = f"{prefix}_tx_count"
        if tx_count_col not in df.columns:
            df[tx_count_col] = df.groupby(entity)[entity].transform("count")
        zscore_col = f"{prefix}{amt_zscore_suffix}"
        if zscore_col in df.columns:
            z_raw = df[zscore_col].abs().fillna(0.0).values
            if "TransactionAmt" in df.columns:
                amt = df["TransactionAmt"].values.astype(float)
                global_z = np.abs((amt - np.mean(amt)) / max(float(np.std(amt)), 1e-6))
            else:
                global_z = np.zeros(n)
            reliable = df[tx_count_col].values >= min_tx_count
            z_final = np.where(reliable, z_raw, global_z)
            entity_scores[f"{entity}_zscore_norm"] = np.clip(z_final, 0, 10) / 10.0
        else:
            entity_scores[f"{entity}_zscore_norm"] = np.zeros(n)
        for suffix, label in [("_velocity_1h", "v1h"), ("_velocity_1d", "v1d")]:
            v_col = f"{prefix}{suffix}"
            if v_col in df.columns:
                arr = df[v_col].fillna(0.0).values.astype(float)
                entity_scores[f"{entity}_{label}_norm"] = np.clip(arr, 0, velocity_cap) / velocity_cap
            else:
                entity_scores[f"{entity}_{label}_norm"] = np.zeros(n)

    entity_weights = {}
    for entity in entity_cols:
        if entity not in df.columns:
            continue
        if target_col and target_col in df.columns:
            global_rate = float(df[target_col].mean())
            fraud_rates = df.groupby(entity)[target_col].mean()
            entity_weights[entity] = float(np.clip(fraud_rates / max(global_rate, 1e-6), 0, 3).mean())
        else:
            entity_weights[entity] = 1.0
    total_weight = sum(entity_weights.values()) or 1.0
    for k in entity_weights:
        entity_weights[k] /= total_weight

    combined = np.zeros(n)
    for entity in entity_cols:
        if entity not in df.columns:
            continue
        w   = entity_weights.get(entity, 1.0 / max(len(entity_cols), 1))
        z   = entity_scores.get(f"{entity}_zscore_norm", np.zeros(n))
        v1h = entity_scores.get(f"{entity}_v1h_norm",   np.zeros(n))
        v1d = entity_scores.get(f"{entity}_v1d_norm",   np.zeros(n))
        combined += (z * 0.5 + v1h * 0.3 + v1d * 0.2) * w

    df["entity_score"] = np.round(np.clip(combined, 0.0, 1.0), 4)
    df["entity_drivers"] = ""
    return df


def detect_temporal_anomalies(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    hour_col: str = "tx_hour",
    is_night_col: str = "tx_is_night",
    is_weekend_col: str = "tx_is_weekend",
    is_business_col: str = "tx_is_business",
    velocity_1h_cols: Optional[list] = None,
    velocity_1d_cols: Optional[list] = None,
    velocity_cap: float = 20.0,
    verbose: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    score_components = pd.DataFrame(index=df.index)
    if hour_col in df.columns:
        if target_col and target_col in df.columns:
            hour_fraud_rate = df.groupby(hour_col)[target_col].mean()
            hour_lift = (hour_fraud_rate / hour_fraud_rate.clip(1e-6, None)).clip(0, 3.0)
            hour_score = df[hour_col].map(hour_lift).fillna(1.0) / 3.0
        else:
            night_hours = set(range(22, 24)) | set(range(0, 6))
            hour_score = df[hour_col].apply(lambda h: 0.7 if h in night_hours else 0.3)
        score_components["hour_score"] = hour_score.clip(0, 1)
    else:
        score_components["hour_score"] = 0.3

    flag_score = pd.Series(0.0, index=df.index)
    if is_night_col in df.columns:
        flag_score += df[is_night_col].fillna(0) * 0.4
    if is_weekend_col in df.columns:
        flag_score += df[is_weekend_col].fillna(0) * 0.2
    if is_business_col in df.columns:
        flag_score -= df[is_business_col].fillna(0) * 0.1
    score_components["flag_score"] = flag_score.clip(0, 1)

    if velocity_1h_cols is None:
        velocity_1h_cols = [c for c in df.columns if c.endswith("_velocity_1h")]
    if velocity_1d_cols is None:
        velocity_1d_cols = [c for c in df.columns if c.endswith("_velocity_1d")]
    all_vel = [c for c in velocity_1h_cols + velocity_1d_cols if c in df.columns]
    if all_vel:
        vel_matrix = df[all_vel].fillna(0).clip(0, velocity_cap) / velocity_cap
        score_components["velocity_score"] = vel_matrix.max(axis=1).clip(0, 1)
    else:
        score_components["velocity_score"] = 0.0

    df["temporal_score"] = (
        score_components["hour_score"] * 0.35
        + score_components["flag_score"] * 0.25
        + score_components["velocity_score"] * 0.40
    ).clip(0, 1).round(4)
    df["temporal_drivers"] = ""
    return df


def run_anomaly_detection(
    df: pd.DataFrame,
    entity_cols: Optional[list] = None,
    target_col: Optional[str] = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    contamination: float = 0.035,
    z_score_cap: float = 10.0,
    velocity_cap: float = 20.0,
    min_tx_count: int = 5,
    verbose: bool = False,
) -> Tuple:
    if entity_cols is None:
        entity_cols = ["card1", "P_emaildomain", "DeviceType"]
    fs_report = feature_selection_report(df=df, target_col=target_col,
                                          variance_threshold=variance_threshold,
                                          correlation_threshold=correlation_threshold,
                                          verbose=verbose)
    df = detect_column_anomalies(df=df, target_col=target_col,
                                  z_score_cap=z_score_cap, verbose=verbose)
    df = detect_multivariate_anomalies(df=df, selected_features=fs_report["selected_features"],
                                        contamination=contamination, verbose=verbose)
    df = detect_entity_anomalies(df=df, entity_cols=entity_cols, target_col=target_col,
                                  min_tx_count=min_tx_count, velocity_cap=velocity_cap,
                                  verbose=verbose)
    df = detect_temporal_anomalies(df=df, target_col=target_col,
                                    velocity_cap=velocity_cap, verbose=verbose)
    explanations = {}
    id_col = "TransactionID" if "TransactionID" in df.columns else None
    for idx, row in df.iterrows():
        tx_id = row.get(id_col, idx) if id_col else idx
        explanations[tx_id] = {
            "column_score_max":   row.get("column_score_max", 0),
            "column_score_mean":  row.get("column_score_mean", 0),
            "multivariate_score": row.get("multivariate_score", 0),
            "entity_score":       row.get("entity_score", 0),
            "temporal_score":     row.get("temporal_score", 0),
        }
    return df, explanations, fs_report