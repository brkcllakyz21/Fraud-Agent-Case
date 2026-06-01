# Fraud Detection Platform

IEEE-CIS Fraud Detection veri seti üzerine inşa edilmiş, uçtan uca açıklanabilir anomali tespit platformu. Adım 1–7'de geliştirilen analiz ve modelleme altyapısı bu platformda bir FastAPI servisi olarak sunulmaktadır.

---

## Mimari Genel Bakış

```
İstemci (HTTP)
      │
      ▼
 FastAPI (Adım 10)
  ├── /score          ──► OrchestratorAgent
  ├── /score/batch    ──► OrchestratorAgent (toplu)
  ├── /explain        ──► OrchestratorAgent → ExplanationAgent
  ├── /rules/evaluate ──► RuleAgent
  └── /rag/query      ──► RAG Pipeline

 OrchestratorAgent (Adım 9)
  ├── ScoringAgent     ← fraud skoru + risk seviyesi
  ├── RuleAgent        ← YAML tabanlı kural motoru (Adım 7)
  └── ExplanationAgent ← RAG + LLM açıklaması (Adım 8)

 PipelineService (Hibrit Cache)
  ├── ScoreCache       ← Adım 7 parquet'ten ID bazlı lookup
  └── PipelineRunner   ← Adım 3–7 canlı çalıştırma (yeni ID)

 dependency_injector Container (Bonus)
  └── Singleton: KB, FAISS, embedding model, tüm agent'lar
```

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| API | FastAPI + Uvicorn |
| Agent Orchestration | Özel BaseAgent ABC + AgentMessage |
| RAG — Embedding | sentence-transformers/all-MiniLM-L6-v2 (CPU) |
| RAG — Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| RAG — LLM | LM Studio (google/gemma-4-e4b, local) |
| Anomali Tespiti | Isolation Forest (scikit-learn) |
| Kural Motoru | YAML tabanlı configurable engine |
| Dependency Injection | dependency-injector |
| Veri | pandas + pyarrow (parquet) |

---

## Klasör Yapısı

```
fraud_platform/
├── knowledge_base/
│   └── fraud_kb.json              # 24 döküman (15 sentetik + 9 AUC bazlı)
├── rag/
│   ├── kb_builder.py              # KB yükleme
│   ├── embedder.py                # sentence-transformers
│   ├── vector_store.py            # FAISS build/save/load/search
│   ├── narrative.py               # transaction → doğal dil (deterministik)
│   ├── prompt_builder.py          # RAG prompt oluşturma
│   ├── llm_client.py              # LM Studio HTTP client
│   └── rag_pipeline.py            # RAG orkestratörü
├── agents/
│   ├── base_agent.py              # BaseAgent ABC + AgentMessage
│   ├── scoring_agent.py           # Fraud skoru + risk seviyesi
│   ├── explanation_agent.py       # RAG tabanlı açıklama
│   ├── rule_agent.py              # Kural değerlendirme
│   └── orchestrator.py            # Sinyal bazlı strateji + task delegation
├── api/
│   ├── main.py                    # FastAPI app + lifespan
│   ├── schemas.py                 # Pydantic modeller
│   ├── dependencies.py            # FastAPI Depends() köprüsü
│   └── routers/
│       ├── score.py               # /score + /score/batch
│       ├── explain.py             # /explain
│       ├── rules.py               # /rules/evaluate
│       └── rag.py                 # /rag/query
├── pipeline/
│   ├── pipeline_service.py        # ScoreCache + PipelineRunner + PipelineService
│   ├── step3_features.py          # Feature engineering
│   ├── step4_anomaly.py           # Anomali tespiti + IF model yükleme
│   ├── step5_scoring.py           # Skor agregasyonu
│   ├── step6_context.py           # Context adjustment
│   └── models/
│       ├── isolation_forest.joblib
│       ├── if_scaler.joblib
│       └── if_feature_list.json
├── rules/
│   └── fraud_rules.yaml           # 10 iş kuralı (YAML)
├── containers.py                  # dependency_injector Container
├── config.py                      # Merkezi ayarlar
├── rule_engine.py                 # Adım 7 rule engine
└── requirements.txt
```

---

## Kurulum

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. LM Studio'yu başlat
#    - google/gemma-4-e4b modelini yükle
#    - Local Server'ı port 1234'te başlat

# 3. config.py'de parquet yolunu ayarla
#    STEP7_OUTPUT_PATH = "outputs/step7/df_rules_train.parquet"

# 4. API'yi başlat (fraud_platform/ içinden)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Startup sırasında otomatik olarak şunlar gerçekleşir:
- Knowledge base (24 döküman) yüklenir
- sentence-transformers embedding modeli yüklenir
- FAISS index yoksa build edilir, varsa diskten yüklenir
- Adım 7 parquet'ten score cache yüklenir
- LM Studio bağlantısı kontrol edilir

---

## API Endpoint'leri

### `GET /health`
Platform sağlık durumu.

```json
{
  "status": "ok",
  "llm_available": true,
  "faiss_loaded": true,
  "kb_doc_count": 24
}
```

---

### `POST /score`
Tek transaction fraud skoru. Hibrit strateji: önce cache, yoksa pipeline.

**Request:**
```json
{
  "transaction": {
    "TransactionID": 2987010,
    "TransactionAmt": 950.0,
    "tx_hour": 23,
    "tx_is_night": true,
    "card1_velocity_1d": 12,
    "card1_amt_zscore": 3.1,
    "column_score_mean": 0.72,
    "entity_score": 0.22
  },
  "mode": "auto"
}
```

**Response:**
```json
{
  "transaction_id": 2987010,
  "fraud_score": 0.78,
  "risk_level": "very_high",
  "strategy": "statistical_new_card",
  "dominant_signals": ["column_score_mean=0.720 (AUC 0.73, weight 37%)"],
  "pipeline_source": "cache",
  "agents_called": ["scoring_agent", "rule_agent", "explanation_agent"]
}
```

**mode değerleri:** `auto` (sinyal bazlı), `score_only`, `score_and_rules`, `full`

**pipeline_source değerleri:** `cache` (Adım 7 parquet), `pipeline` (Adım 3–7 çalıştı), `fallback` (ham değerler kullanıldı)

**risk_level değerleri:** `low` (<0.3), `medium` (0.3–0.5), `high` (0.5–0.7), `very_high` (>0.7)

---

### `POST /score/batch`
Toplu transaction scoring. Max 1000 transaction.

```json
{
  "transactions": [ { "TransactionID": 1001, ... }, ... ],
  "mode": "auto"
}
```

```json
{
  "total": 3, "processed": 3, "failed": 0,
  "risk_summary": { "low": 1, "medium": 0, "high": 1, "very_high": 1 },
  "results": [
    { "transaction_id": 1001, "fraud_score": 0.78, "risk_level": "very_high",
      "strategy": "statistical_new_card" }
  ]
}
```

---

### `POST /explain`
RAG + LLM tabanlı fraud açıklaması.

**Response:**
```json
{
  "fraud_score": 0.78,
  "strategy": "statistical_new_card",
  "dominant_signals": ["column_score_mean=0.720 (AUC 0.73, weight 37%)"],
  "narrative": "Detection strategy: statistical deviation with low entity history...",
  "driver_features": ["high_column_score_mean", "nighttime_transaction", ...],
  "retrieved_docs": [
    { "id": "KB024", "title": "Weak Entity Signal with Strong Column Signal", "score": 0.91 }
  ],
  "explanation": "**Summary:** Transaction flagged due to multiple severe anomalies...",
  "rule_audit_trail": [ { "rule_id": "R001", "triggered": true, "multiplier": 1.5 } ]
}
```

---

### `POST /rules/evaluate`
YAML kural motoru değerlendirmesi. 10 kuralın tamamı için audit trail.

```json
{
  "rules_triggered": ["R001", "R005", "R008", "R009"],
  "max_multiplier": 1.5,
  "severity": "HIGH",
  "rule_audit_trail": [
    { "rule_id": "R001", "triggered": true, "multiplier": 1.5, "severity": "HIGH" },
    { "rule_id": "R002", "triggered": false, "multiplier": 1.45, "severity": "HIGH" }
  ]
}
```

---

### `POST /rag/query`
Serbest metin ile knowledge base sorgusu.

```json
{ "query": "nighttime high velocity transaction", "top_k": 3 }
```

```json
{
  "retrieved_docs": [ { "id": "KB002", "title": "Nighttime High-Value Transaction Policy" } ],
  "answer": "**Summary:** ... **Recommendation:** review"
}
```

---

## Orchestrator Strateji Sistemi

OrchestratorAgent, Adım 5 AUC ağırlıklarını kullanarak hangi anomali katmanının baskın olduğunu tespit eder ve buna göre agent delegasyonu yapar.

| Strateji | Tetikleme Koşulu | Agent'lar |
|----------|-----------------|-----------|
| `statistical` | column_score_mean > 0.60 | Scoring + Rules + Explanation |
| `statistical_new_card` | column yüksek + entity < 0.30 | Scoring + Rules + Explanation |
| `behavioral` | entity_score > 0.50 | Scoring + Rules + Explanation |
| `behavioral_multivariate` | entity + mv yüksek | Scoring + Rules + Explanation |
| `multivariate` | multivariate_score > 0.55 | Scoring + Rules + Explanation |
| `temporal` | temporal_score > 0.50 | Scoring + Rules |
| `rule_based` | model sinyalleri zayıf | Scoring (+ Rules eşikte) |

---

## Knowledge Base

24 döküman, 3 grupta:

**Orijinal (KB001–KB015):** Sentetik fraud kuralları — velocity, gece, email domain, kart tipi, güvenilir entity vb.

**Model sinyal dökümanları (KB016–KB021):** Adım 5 AUC bulgularına dayalı. column_score_mean (AUC 0.73, %37 ağırlık), column_score_max (AUC 0.70, %33), multivariate (AUC 0.64, %23), entity (AUC 0.53), temporal (AUC 0.52).

**Feature kombinasyon dökümanları (KB022–KB024):** Adım 2 analizinden türetilmiş. Yüksek riskli kombinasyonlar: column+gece+velocity, amount zscore+column dual anomaly, yeni kart pattern (düşük entity + yüksek column).

---

## Hibrit Pipeline Akışı

```
/score isteği gelir
      │
      ▼
TransactionID cache'de var mı? (Adım 7 parquet)
      │ evet                    │ hayır
      ▼                         ▼
Skorları parquet'ten al    Adım 3–7 çalıştır
pipeline_source: "cache"   pipeline_source: "pipeline"
      │                         │
      └─────────┬───────────────┘
                ▼
       OrchestratorAgent
       Strateji tespit et
       Agent'ları tetikle
```

---

## Dependency Injection (Bonus)

`containers.py` — dependency_injector ile 5 tasarım deseni:

| Desen | Uygulama |
|-------|----------|
| Singleton | FAISS index, embedding model, agent'lar — bir kez oluşturulur |
| Factory | AgentMessage — her çağrıda yeni |
| Configuration | Tüm ayarlar config.settings'ten, test'te override edilebilir |
| Strategy | llm_model değiştirilerek farklı LLM'e geçiş |
| Repository | kb_docs — knowledge base için repository rolü |

---

## Notlar

- **LLM (Gemma 4E4B):** Thinking modeli olduğundan ilk inference ~2–3 dakika sürebilir. Yanıt kalitesi yüksek, reasoning transparent.
- **FAISS index:** Startup'ta otomatik build edilir. KB güncellenirse `knowledge_base/faiss_index.*` dosyalarını silerek yeniden build edin.
- **Isolation Forest:** `pipeline/models/` altındaki joblib dosyaları Adım 4 notebook'u çalıştırıldığında otomatik güncellenir.
- **Dataları Doğrudan Ana Klasör Altına Çıkararak Kodların Çalışmasını Sağlayabilirsiniz**