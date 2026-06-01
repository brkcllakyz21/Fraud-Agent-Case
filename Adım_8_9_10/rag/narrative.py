from __future__ import annotations

from typing import Any, Dict, List


def transaction_to_narrative(row: Dict[str, Any]) -> str:
    parts = []

    # ── Strateji context (orchestrator'dan gelir) ───────────────────────────
    strategy_desc = row.get("_strategy_description", "")
    dominant_text = row.get("_dominant_signals_text", "")
    if strategy_desc:
        parts.append(f"Detection strategy: {strategy_desc}")
    if dominant_text:
        parts.append(f"Dominant signals: {dominant_text}.")

    # Tutar
    amt = row.get("TransactionAmt")
    if amt is not None:
        parts.append(f"Transaction amount: ${float(amt):.2f}.")

    # Zaman
    hour = row.get("tx_hour")
    if hour is not None:
        is_night = row.get("tx_is_night", hour >= 22 or hour < 6)
        is_weekend = row.get("tx_is_weekend", False)
        time_desc = f"Occurred at hour {int(hour):02d}:00"
        if is_night:
            time_desc += " (nighttime)"
        if is_weekend:
            time_desc += ", on a weekend"
        parts.append(time_desc + ".")

    # ── Ensemble skorları (AUC ağırlıklı, en güçlü sinyaller) ──────────────
    col_mean = row.get("column_score_mean")
    col_max = row.get("column_score_max")
    mv_score = row.get("multivariate_score")
    entity_score = row.get("entity_score")
    temporal_score = row.get("temporal_score")

    if col_mean is not None:
        v = float(col_mean)
        label = "very high" if v > 0.7 else ("high" if v > 0.5 else "normal")
        parts.append(
            f"Column anomaly score (AUC 0.73, weight 37.1%): {v:.4f} ({label}) — "
            f"measures aggregate statistical deviation across all features."
        )

    if col_max is not None:
        v = float(col_max)
        label = "very high" if v > 0.65 else ("elevated" if v > 0.5 else "normal")
        parts.append(
            f"Max column deviation score (AUC 0.70, weight 32.5%): {v:.4f} ({label}) — "
            f"captures worst-case single feature anomaly."
        )

    if mv_score is not None:
        v = float(mv_score)
        label = "elevated" if v > 0.5 else "normal"
        parts.append(
            f"Multivariate Isolation Forest score (AUC 0.64, weight 22.7%): {v:.4f} ({label})."
        )

    if entity_score is not None:
        v = float(entity_score)
        label = "anomalous" if v > 0.5 else "within baseline"
        parts.append(
            f"Entity behavioral score (AUC 0.53, weight 4.85%): {v:.4f} ({label})."
        )

    if temporal_score is not None:
        v = float(temporal_score)
        label = "anomalous" if v > 0.5 else "normal"
        parts.append(
            f"Temporal pattern score (AUC 0.52, weight 2.86%): {v:.4f} ({label})."
        )

    # ── Entity davranış sinyalleri ──────────────────────────────────────────
    velocity = row.get("card1_velocity_1d")
    if velocity is not None:
        v = float(velocity)
        label = "high" if v > 6 else "normal"
        parts.append(f"Card daily velocity: {v:.0f} transactions ({label}).")

    zscore = row.get("card1_amt_zscore")
    if zscore is not None:
        z = float(zscore)
        label = "elevated" if z > 2 else "normal"
        parts.append(f"Amount Z-score vs card history: {z:.2f} ({label}).")

    # ── Fraud lift sinyalleri ───────────────────────────────────────────────
    for col, name in [
        ("ctx_card4_fraud_lift", "Card network (card4)"),
        ("ctx_card6_fraud_lift", "Card type (card6)"),
        ("ctx_pemaildomain_fraud_lift", "Email domain"),
        ("ctx_productcd_fraud_lift", "Product category"),
    ]:
        val = row.get(col)
        if val is not None:
            v = float(val)
            if v > 1.2:
                parts.append(f"{name} fraud lift: {v:.2f} (elevated risk).")
            elif v < 0.8:
                parts.append(f"{name} fraud lift: {v:.2f} (low risk).")

    # ── Tetiklenen kurallar ─────────────────────────────────────────────────
    rules_triggered = row.get("rules_triggered", "")
    rule_severity = row.get("rule_severity", "")
    if rules_triggered:
        parts.append(f"Rules triggered: {rules_triggered} (severity: {rule_severity}).")

    # ── Final skor ─────────────────────────────────────────────────────────
    final_score = (
        row.get("rule_adjusted_score")
        or row.get("context_adjusted_score")
        or row.get("fraud_score")
    )
    if final_score is not None:
        parts.append(f"Final fraud score: {float(final_score):.4f}.")

    return " ".join(parts) if parts else "No transaction signals available."


def extract_driver_features(row: Dict[str, Any]) -> List[str]:
    drivers = []

    # ── Ensemble sinyal driverları (en ağırlıklılar önce) ──────────────────
    col_mean = row.get("column_score_mean")
    if col_mean is not None and float(col_mean) > 0.6:
        drivers.append(f"high_column_score_mean:{float(col_mean):.2f}_auc0.73_weight37pct")

    col_max = row.get("column_score_max")
    if col_max is not None and float(col_max) > 0.5:
        drivers.append(f"high_column_score_max:{float(col_max):.2f}_auc0.70_weight33pct")

    mv = row.get("multivariate_score")
    if mv is not None and float(mv) > 0.5:
        drivers.append(f"elevated_multivariate_score:{float(mv):.2f}_isolation_forest")

    entity = row.get("entity_score")
    if entity is not None and float(entity) > 0.5:
        drivers.append("entity_behavioral_anomaly")
    elif entity is not None and float(entity) < 0.3 and col_mean is not None and float(col_mean) > 0.6:
        drivers.append("new_card_pattern_low_entity_high_column")

    temporal = row.get("temporal_score")
    if temporal is not None and float(temporal) > 0.5:
        drivers.append("temporal_pattern_anomaly")

    # ── Zaman/davranış driverları ───────────────────────────────────────────
    hour = row.get("tx_hour", -1)
    is_night = row.get("tx_is_night", hour >= 22 or (isinstance(hour, (int, float)) and hour < 6))
    if is_night:
        drivers.append("nighttime_transaction")

    if row.get("tx_is_weekend"):
        drivers.append("weekend_transaction")

    velocity = row.get("card1_velocity_1d")
    if velocity is not None and float(velocity) > 6:
        drivers.append("high_velocity")

    zscore = row.get("card1_amt_zscore")
    if zscore is not None and float(zscore) > 2:
        drivers.append("high_amount_zscore")

    # ── Dual anomaly: hem entity hem column yüksek ─────────────────────────
    if (
        zscore is not None and float(zscore) > 2.0
        and col_max is not None and float(col_max) > 0.65
    ):
        drivers.append("dual_amount_anomaly_entity_and_population")

    amt = row.get("TransactionAmt")
    if amt is not None and float(amt) >= 2000:
        drivers.append("extreme_transaction_amount")

    for col, tag in [
        ("ctx_card4_fraud_lift", "risky_card_network"),
        ("ctx_card6_fraud_lift", "risky_card_type"),
        ("ctx_pemaildomain_fraud_lift", "risky_email_domain"),
        ("ctx_productcd_fraud_lift", "risky_product_category"),
    ]:
        val = row.get(col)
        if val is not None and float(val) > 1.2:
            drivers.append(tag)

    score = row.get("rule_adjusted_score") or row.get("fraud_score", 0)
    if score is not None and float(score) > 0.7:
        drivers.append("very_high_fraud_score")
    elif score is not None and float(score) > 0.5:
        drivers.append("high_fraud_score")

    return drivers