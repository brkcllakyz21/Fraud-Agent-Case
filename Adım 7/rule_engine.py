"""
Adım 7 — Rule Engine
====================
YAML tabanlı, configurable, audit-trail destekli fraud rule engine.

Tasarım prensipleri:
- İşaretleyici + düzeltici: kural silmez, multiplier uygular
- Conflict resolution: max_multiplier (tüm kurallar çalışır, en yüksek multiplier kazanır)
- adjusted_score × max_multiplier → rule_adjusted_score  (Adım 6 ile zincirleme)
- rule_score: [0,1] normalize edilmiş tetiklenen kural sayısı
- Her satır için tam audit trail (hangi kural, ne multiplier, neden)
"""

from __future__ import annotations

import json
import operator
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Operator map
# ---------------------------------------------------------------------------
OPS: Dict[str, Any] = {
    "==":  operator.eq,
    "!=":  operator.ne,
    ">":   operator.gt,
    ">=":  operator.ge,
    "<":   operator.lt,
    "<=":  operator.le,
}


# ---------------------------------------------------------------------------
# 1. Config loader
# ---------------------------------------------------------------------------
def load_rules(path: Union[str, Path]) -> dict:
    """
    YAML veya JSON kural dosyasını yükler ve doğrular.

    Returns
    -------
    dict  — meta + rules listesi
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kural dosyası bulunamadı: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            config = yaml.safe_load(f)
        elif path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Desteklenmeyen format: {path.suffix} (yaml/json bekleniyor)")

    assert "rules" in config, "Config'de 'rules' anahtarı bulunamadı."
    enabled = [r for r in config["rules"] if r.get("enabled", True)]
    print(f"[RuleEngine] {len(enabled)} / {len(config['rules'])} kural yüklendi.")
    return config


# ---------------------------------------------------------------------------
# 2. Tek kural evaluator
# ---------------------------------------------------------------------------
def _evaluate_condition(row: pd.Series, check: dict) -> bool:
    """
    Tek bir check dict'ini satır üzerinde değerlendirir.
    """
    field = check["field"]
    op_fn = OPS.get(check["op"])
    if op_fn is None:
        raise ValueError(f"Bilinmeyen operatör: {check['op']}")
    if field not in row.index:
        return False
    field_val = row[field]
    if pd.isna(field_val):
        return False
    return bool(op_fn(field_val, check["value"]))


def evaluate_rule(row: pd.Series, rule: dict) -> Tuple[bool, float, str]:
    """
    Tek kuralı değerlendirir.

    Returns
    -------
    (triggered, multiplier, explanation)
    """
    cond = rule["conditions"]
    logic = cond.get("logic", "AND").upper()
    checks = cond["checks"]

    results = [_evaluate_condition(row, c) for c in checks]

    if logic == "AND":
        triggered = all(results)
    elif logic == "OR":
        triggered = any(results)
    else:
        raise ValueError(f"Bilinmeyen logic: {logic}")

    if triggered:
        return True, rule["action"]["multiplier"], rule["action"]["explanation"]
    return False, 1.0, ""


# ---------------------------------------------------------------------------
# 3. Batch evaluator
# ---------------------------------------------------------------------------
def evaluate_all_rules(
    df: pd.DataFrame,
    config: dict,
    score_col: str = "adjusted_score",
) -> pd.DataFrame:
    """
    Tüm etkin kuralları DataFrame üzerinde çalıştırır.

    Conflict resolution: max_multiplier
        → Her satır için tetiklenen kurallar içinde en yüksek multiplier seçilir.

    Yeni kolonlar
    -------------
    rule_adjusted_score  : adjusted_score × max_multiplier, [0,1] clip
    rule_score           : tetiklenen kural sayısı / toplam kural (normalize)
    rules_triggered      : tetiklenen kural ID'leri (pipe-separated string)
    rules_max_multiplier : kazanan multiplier değeri
    rule_explanation     : kazanan kuralın açıklaması
    rule_severity        : kazanan kuralın severity etiketi
    rule_flags           : tetiklenen tüm flag'ler (pipe-separated string)
    rule_audit_trail     : her tetiklenen kural için detay (JSON string)
    """
    if score_col not in df.columns:
        raise KeyError(f"'{score_col}' kolonu bulunamadı. Adım 6 çıktısını kontrol edin.")

    enabled_rules = [r for r in config["rules"] if r.get("enabled", True)]
    n_rules = len(enabled_rules)
    conflict_strategy = config.get("meta", {}).get("conflict_resolution", "max_multiplier")

    print(f"[RuleEngine] {n_rules} kural, strateji: {conflict_strategy}")
    print(f"[RuleEngine] Satır sayısı: {len(df):,}")

    max_multipliers = np.ones(len(df))
    triggered_counts = np.zeros(len(df), dtype=int)
    triggered_ids = [[] for _ in range(len(df))]
    triggered_flags = [[] for _ in range(len(df))]
    winning_explanations = [""] * len(df)
    winning_severities = ["NONE"] * len(df)
    audit_trails = [[] for _ in range(len(df))]

    for rule in enabled_rules:
        rid = rule["id"]
        print(f"  -> {rid}: {rule['name']} ...", end=" ")

        rule_results = df.apply(lambda row, r=rule: evaluate_rule(row, r), axis=1)

        triggered_mask = rule_results.apply(lambda x: x[0])
        multipliers_series = rule_results.apply(lambda x: x[1])
        explanations_series = rule_results.apply(lambda x: x[2])

        n_triggered = triggered_mask.sum()
        print(f"{n_triggered:,} satir tetiklendi")

        for idx_pos, (idx, trig) in enumerate(triggered_mask.items()):
            if not trig:
                continue
            mult = multipliers_series[idx]
            expl = explanations_series[idx]

            triggered_counts[idx_pos] += 1
            triggered_ids[idx_pos].append(rid)
            triggered_flags[idx_pos].append(rule["action"]["rule_flag"])

            audit_trails[idx_pos].append({
                "rule_id": rid,
                "rule_name": rule["name"],
                "multiplier": mult,
                "severity": rule["action"]["severity"],
                "flag": rule["action"]["rule_flag"],
            })

            # Max multiplier conflict resolution
            if mult > max_multipliers[idx_pos]:
                max_multipliers[idx_pos] = mult
                winning_explanations[idx_pos] = expl.strip()
                winning_severities[idx_pos] = rule["action"]["severity"]
            # Güvenlik kuralı (multiplier < 1): sadece hiç yükseltici kural yoksa uygula
            elif mult < 1.0 and max_multipliers[idx_pos] == 1.0:
                max_multipliers[idx_pos] = mult
                winning_explanations[idx_pos] = expl.strip()
                winning_severities[idx_pos] = rule["action"]["severity"]

    df = df.copy()
    df["rules_max_multiplier"] = max_multipliers
    df["rule_adjusted_score"] = (df[score_col] * max_multipliers).clip(0.0, 1.0)
    df["rule_score"] = triggered_counts / n_rules
    df["rules_triggered"] = ["|".join(ids) if ids else "" for ids in triggered_ids]
    df["rule_flags"] = ["|".join(flags) if flags else "" for flags in triggered_flags]
    df["rule_explanation"] = winning_explanations
    df["rule_severity"] = winning_severities
    df["rule_audit_trail"] = [
        json.dumps(trail, ensure_ascii=False) if trail else "[]"
        for trail in audit_trails
    ]

    return df


# ---------------------------------------------------------------------------
# 4. Explainability raporu
# ---------------------------------------------------------------------------
def generate_rule_report(
    df: pd.DataFrame,
    config: dict,
    target_col: Optional[str] = "isFraud",
) -> dict:
    """
    Rule engine sonuçlarını özetleyen rapor üretir.
    """
    enabled_rules = [r for r in config["rules"] if r.get("enabled", True)]
    rule_stats = []

    for rule in enabled_rules:
        flag = rule["action"]["rule_flag"]
        mask = df["rule_flags"].str.contains(flag, na=False)
        n = mask.sum()

        stat = {
            "rule_id": rule["id"],
            "rule_name": rule["name"],
            "priority": rule["priority"],
            "severity": rule["action"]["severity"],
            "multiplier": rule["action"]["multiplier"],
            "n_triggered": int(n),
            "pct_triggered": round(n / len(df) * 100, 2),
        }

        if target_col and target_col in df.columns and n > 0:
            fraud_rate = df.loc[mask, target_col].mean()
            overall_rate = df[target_col].mean()
            stat["fraud_rate"] = round(float(fraud_rate), 4)
            stat["fraud_lift"] = round(float(fraud_rate / overall_rate) if overall_rate > 0 else 0, 3)
        else:
            stat["fraud_rate"] = None
            stat["fraud_lift"] = None

        rule_stats.append(stat)

    sep_report = {}
    if target_col and target_col in df.columns:
        fraud_mask = df[target_col] == 1
        for col in ["adjusted_score", "rule_adjusted_score"]:
            if col in df.columns:
                f_mean = df.loc[fraud_mask, col].mean()
                nf_mean = df.loc[~fraud_mask, col].mean()
                sep_report[col] = {
                    "fraud_mean": round(float(f_mean), 4),
                    "nonfraud_mean": round(float(nf_mean), 4),
                    "separation_ratio": round(float(f_mean / nf_mean) if nf_mean > 0 else 0, 4),
                }

    severity_dist = df["rule_severity"].value_counts().to_dict()

    n_rules_dist = df["rules_triggered"].apply(
        lambda x: len(x.split("|")) if x else 0
    ).value_counts().sort_index().to_dict()

    report = {
        "meta": {
            "total_rows": len(df),
            "total_rules": len(enabled_rules),
            "conflict_resolution": config.get("meta", {}).get("conflict_resolution"),
        },
        "rule_stats": rule_stats,
        "separation_metrics": sep_report,
        "severity_distribution": {str(k): int(v) for k, v in severity_dist.items()},
        "rules_triggered_per_row_dist": {str(k): int(v) for k, v in n_rules_dist.items()},
    }

    return report


# ---------------------------------------------------------------------------
# 5. Orkestratör
# ---------------------------------------------------------------------------
def run_rule_engine(
    df: pd.DataFrame,
    rules_path: Union[str, Path],
    score_col: str = "adjusted_score",
    target_col: Optional[str] = "isFraud",
    output_dir: Union[str, Path] = "outputs/step7",
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Rule engine ana orkestratörü.

    Akış
    ----
    1. YAML yükle -> doğrula
    2. Tüm kuralları değerlendir (max_multiplier conflict resolution)
    3. Explainability raporu üret
    4. Çıktıları kaydet (parquet + JSON)

    Returns
    -------
    (df_rules, rule_report)
    """
    print("=" * 60)
    print("ADIM 7 — RULE ENGINE")
    print("=" * 60)

    config = load_rules(rules_path)
    df_rules = evaluate_all_rules(df, config, score_col=score_col)
    rule_report = generate_rule_report(df_rules, config, target_col=target_col)

    if save_outputs:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        df_rules.to_parquet(out / "df_rules_train.parquet", index=False)
        with open(out / "rule_report_train.json", "w", encoding="utf-8") as f:
            json.dump(rule_report, f, ensure_ascii=False, indent=2)

        print(f"\n[RuleEngine] Çıktılar kaydedildi: {out}")

    print("\n--- SEPARATION METRIKLERI ---")
    for col, metrics in rule_report.get("separation_metrics", {}).items():
        print(
            f"  {col:25s} | fraud={metrics['fraud_mean']:.4f} "
            f"nonfraud={metrics['nonfraud_mean']:.4f} "
            f"ratio={metrics['separation_ratio']:.4f}x"
        )

    print("\n--- KURAL OZETI (fraud lift sirali) ---")
    stats = sorted(
        rule_report["rule_stats"],
        key=lambda x: x.get("fraud_lift") or 0,
        reverse=True,
    )
    for s in stats:
        lift_str = f"lift={s['fraud_lift']:.3f}" if s["fraud_lift"] is not None else "lift=N/A"
        print(
            f"  {s['rule_id']} [{s['severity']:6s}] "
            f"n={s['n_triggered']:6,} ({s['pct_triggered']:5.1f}%) "
            f"{lift_str:12s} mult={s['multiplier']:.2f}  {s['rule_name']}"
        )

    print("\n--- SEVERITY DAGILIMI ---")
    for sev, cnt in rule_report["severity_distribution"].items():
        print(f"  {sev:8s}: {cnt:,}")

    print("\n--- AYNI SATIRDA TETIKLENEN KURAL SAYISI ---")
    for k, v in sorted(
        rule_report["rules_triggered_per_row_dist"].items(),
        key=lambda x: int(x[0])
    ):
        print(f"  {k} kural: {v:,} satir")

    return df_rules, rule_report