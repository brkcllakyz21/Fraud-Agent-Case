"""
Fraud Detection Platform — API Test Script
Çalıştır: python test_api.py
"""
import json
import requests

BASE_URL = "http://localhost:8000"

# ── Test transaction'ları ──────────────────────────────────────────────────

# Cache hit senaryosu — Adım 7 parquet'te var olan bir ID
# (kendi parquet'indeki gerçek bir ID ile değiştir)
CACHED_TX = {
    "TransactionID": 2987010,        # ← Adım 7 parquet'te var olan bir ID
    "TransactionAmt": 950.0,
    "rule_adjusted_score": 0.78,
    "context_adjusted_score": 0.65,
    "fraud_score": 0.60,
    "tx_hour": 23,
    "tx_is_night": True,
    "tx_is_weekend": False,
    "card1_velocity_1d": 12,
    "card1_amt_zscore": 3.1,
    "ctx_pemaildomain_fraud_lift": 1.8,
    "ctx_productcd_fraud_lift": 1.3,
    "column_score_mean": 0.72,
    "column_score_max": 0.68,
    "multivariate_score": 0.61,
    "entity_score": 0.22,
}

# Pipeline fallback senaryosu — hiç olmayan yeni bir ID
# Pipeline Adım 3-7'yi tetikler (step*.py dosyaları varsa)
NEW_TX = {
    "TransactionID": 9999999,        # ← parquet'te kesinlikle olmayan ID
    "TransactionAmt": 850.0,
    "TransactionDT": 86400,          # pipeline için gerekli (temporal features)
    "ProductCD": "W",
    "card1": 12345,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "addr1": 315,
    "tx_hour": 2,
    "tx_is_night": True,
    "tx_is_weekend": False,
    "card1_velocity_1d": 9,
    "card1_amt_zscore": 2.8,
    "ctx_pemaildomain_fraud_lift": 1.5,
    "ctx_productcd_fraud_lift": 1.2,
}

# Yüksek riskli test transaction (manuel skorlu)
HIGH_RISK_TX = {
    "TransactionID": 99001,
    "TransactionAmt": 950.0,
    "rule_adjusted_score": 0.78,
    "context_adjusted_score": 0.65,
    "fraud_score": 0.60,
    "tx_hour": 23,
    "tx_is_night": True,
    "tx_is_weekend": False,
    "card1_velocity_1d": 12,
    "card1_amt_zscore": 3.1,
    "ctx_pemaildomain_fraud_lift": 1.8,
    "ctx_productcd_fraud_lift": 1.3,
    "column_score_mean": 0.72,
    "column_score_max": 0.68,
    "multivariate_score": 0.61,
    "entity_score": 0.22,
}

# Düşük riskli test transaction
LOW_RISK_TX = {
    "TransactionID": 99002,
    "TransactionAmt": 45.0,
    "rule_adjusted_score": 0.12,
    "context_adjusted_score": 0.10,
    "fraud_score": 0.08,
    "tx_hour": 14,
    "tx_is_night": False,
    "tx_is_weekend": False,
    "card1_velocity_1d": 2,
    "card1_amt_zscore": 0.3,
    "column_score_mean": 0.15,
    "column_score_max": 0.18,
    "entity_score": 0.10,
}


# ── Test fonksiyonları ─────────────────────────────────────────────────────

def sep(title):
    print("\n" + "="*55)
    print(f"TEST: {title}")
    print("="*55)


def test_health():
    sep("/health")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))


def test_score(tx, label=""):
    sep(f"/score  [{label}]")
    r = requests.post(f"{BASE_URL}/score", json={"transaction": tx, "mode": "auto"})
    print(f"Status: {r.status_code}")
    data = r.json()
    print(json.dumps(data, indent=2))


def test_score_cache_hit():
    sep("/score  [CACHE HIT — bilinen TransactionID]")
    r = requests.post(f"{BASE_URL}/score", json={"transaction": CACHED_TX, "mode": "auto"})
    print(f"Status: {r.status_code}")
    data = r.json()
    print(json.dumps(data, indent=2))
    source = data.get("pipeline_source", "unknown")
    if source == "cache":
        print("\n✅ CACHE HIT: Skor Adım 7 parquet'ten geldi.")
    elif source == "pipeline":
        print("\n⚙️  PIPELINE: Skor Adım 3-7 çalıştırılarak üretildi.")
    else:
        print(f"\nℹ️  Kaynak: {source}")


def test_score_pipeline_fallback():
    sep("/score  [PIPELINE FALLBACK — yeni TransactionID=9999999]")
    print("(Adım 3-7 pipeline tetikleniyor, biraz sürebilir...)")
    r = requests.post(
        f"{BASE_URL}/score",
        json={"transaction": NEW_TX, "mode": "auto"},
        timeout=300,
    )
    print(f"Status: {r.status_code}")
    data = r.json()
    print(json.dumps(data, indent=2))
    source = data.get("pipeline_source", "unknown")
    if source == "pipeline":
        print("\n✅ PIPELINE FALLBACK: Yeni ID için Adım 3-7 başarıyla çalıştı.")
    elif source == "cache":
        print("\n⚠️  Cache hit — bu ID parquet'te varmış, farklı bir ID dene.")
    elif source == "error":
        print("\n❌ Pipeline başarısız — step*.py dosyaları eksik olabilir.")
    else:
        print(f"\nℹ️  Kaynak: {source}")


def test_rules(tx, label=""):
    sep(f"/rules/evaluate  [{label}]")
    r = requests.post(f"{BASE_URL}/rules/evaluate", json={"transaction": tx})
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))


def test_rules_audit():
    sep("/rules/evaluate  [FULL AUDIT TRAIL]")
    r = requests.post(f"{BASE_URL}/rules/evaluate", json={"transaction": HIGH_RISK_TX})
    data = r.json()
    print(f"Status: {r.status_code}")
    print(f"Rules triggered : {data.get('rules_triggered')}")
    print(f"Max multiplier  : {data.get('max_multiplier')}")
    print(f"Severity        : {data.get('severity')}")
    print(f"\nAudit trail ({len(data.get('rule_audit_trail', []))} rules):")
    for entry in data.get("rule_audit_trail", []):
        status = "✅ TRIGGERED" if entry["triggered"] else "⬜ skipped  "
        print(f"  {entry['rule_id']} | {status} | x{entry['multiplier']:.2f} | {entry.get('severity', '')}")


def test_explain(tx, label=""):
    sep(f"/explain  [{label}]")
    print("(LLM çağrısı — sürebilir...)")
    r = requests.post(f"{BASE_URL}/explain", json={"transaction": tx}, timeout=300)
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Fraud Score      : {data.get('fraud_score')}")
    print(f"Strategy         : {data.get('strategy')}")
    print(f"Dominant Signals : {data.get('dominant_signals')}")
    print(f"Narrative        : {data.get('narrative')}")
    print(f"Drivers          : {data.get('driver_features')}")
    print(f"Retrieved KB     : {[d['id'] + ' - ' + d['title'] for d in data.get('retrieved_docs', [])]}")
    audit = data.get("rule_audit_trail", [])
    if audit:
        triggered = [e for e in audit if e["triggered"]]
        print(f"Audit (triggered): {[e['rule_id'] for e in triggered]}")
    print(f"\nLLM Explanation:\n{data.get('explanation')}")


def test_rag_query():
    sep("/rag/query")
    r = requests.post(
        f"{BASE_URL}/rag/query",
        json={"query": "nighttime high velocity transaction with risky email domain", "top_k": 3},
        timeout=300,
    )
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Retrieved: {[d['id'] for d in data.get('retrieved_docs', [])]}")
    print(f"\nAnswer:\n{data.get('answer')}")


def test_batch_score():
    sep("/score/batch")
    transactions = [
        {
            "TransactionID": 1001,
            "TransactionAmt": 950.0,
            "rule_adjusted_score": 0.78,
            "tx_hour": 23, "tx_is_night": True,
            "card1_velocity_1d": 12, "card1_amt_zscore": 3.1,
            "column_score_mean": 0.72, "column_score_max": 0.68,
            "multivariate_score": 0.61, "entity_score": 0.22,
        },
        {
            "TransactionID": 1002,
            "TransactionAmt": 45.0,
            "rule_adjusted_score": 0.12,
            "tx_hour": 14, "tx_is_night": False,
            "card1_velocity_1d": 2, "card1_amt_zscore": 0.3,
            "column_score_mean": 0.15, "column_score_max": 0.18,
        },
        {
            "TransactionID": 1003,
            "TransactionAmt": 320.0,
            "rule_adjusted_score": 0.55,
            "tx_hour": 2, "tx_is_night": True,
            "card1_velocity_1d": 5, "card1_amt_zscore": 1.8,
            "column_score_mean": 0.63, "column_score_max": 0.58,
            "entity_score": 0.51, "multivariate_score": 0.57,
        },
    ]
    r = requests.post(
        f"{BASE_URL}/score/batch",
        json={"transactions": transactions, "mode": "auto"},
        timeout=120,
    )
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Total: {data['total']} | Processed: {data['processed']} | Failed: {data['failed']}")
    print(f"Risk summary: {data['risk_summary']}")
    for item in data["results"]:
        print(f"  TX {item['transaction_id']}: score={item['fraud_score']:.2f} | "
              f"{item['risk_level']:10s} | strategy={item.get('strategy')}")


if __name__ == "__main__":
    # ── Temel sistem testi ─────────────────────────────────────────
    test_health()

    # ── Scoring testleri ───────────────────────────────────────────
    test_score(HIGH_RISK_TX, "HIGH RISK — manuel skorlu")
    test_score(LOW_RISK_TX,  "LOW RISK — manuel skorlu")

    # ── Hibrit pipeline testleri ───────────────────────────────────
    test_score_cache_hit()          # Cache hit — bilinen ID
    test_score_pipeline_fallback()  # Pipeline fallback — ID=9999999

    # ── Kural motoru testleri ──────────────────────────────────────
    test_rules(HIGH_RISK_TX, "HIGH RISK")
    test_rules_audit()

    # ── Batch scoring ──────────────────────────────────────────────
    test_batch_score()

    # ── Explainability & RAG ───────────────────────────────────────
    test_explain(HIGH_RISK_TX, "HIGH RISK")
    test_rag_query()