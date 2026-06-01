from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── /score ───────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    transaction: Dict[str, Any] = Field(
        ..., description="Transaction fields (TransactionAmt, tx_hour, vb.)"
    )
    mode: str = Field(
        default="auto",
        description="auto | score_only | score_and_rules | full"
    )


class ScoreBreakdown(BaseModel):
    column_score_mean: Optional[float] = None
    column_score_max: Optional[float] = None
    multivariate_score: Optional[float] = None
    entity_score: Optional[float] = None
    temporal_score: Optional[float] = None
    context_adjusted_score: Optional[float] = None
    rule_adjusted_score: Optional[float] = None


class ScoreResponse(BaseModel):
    transaction_id: Optional[Any] = None
    fraud_score: float
    risk_level: str = Field(..., description="low | medium | high | very_high")
    score_breakdown: Optional[ScoreBreakdown] = None
    strategy: Optional[str] = Field(None, description="Tetiklenen orchestrator stratejisi")
    dominant_signals: List[str] = Field(default_factory=list)
    agents_called: List[str] = Field(default_factory=list)
    agent_errors: Dict[str, str] = Field(default_factory=dict)
    pipeline_source: Optional[str] = Field(None, description="cache | pipeline | error")


# ─── /score/batch ─────────────────────────────────────────────────────────────

class BatchScoreRequest(BaseModel):
    transactions: List[Dict[str, Any]] = Field(
        ..., description="Transaction listesi, max 1000",
        max_length=1000,
    )
    mode: str = Field(
        default="score_only",
        description="auto | score_only | score_and_rules | full — batch'te default score_only"
    )


class BatchScoreItem(BaseModel):
    transaction_id: Optional[Any] = None
    fraud_score: float
    risk_level: str
    strategy: Optional[str] = None
    agent_errors: Dict[str, str] = Field(default_factory=dict)


class BatchScoreResponse(BaseModel):
    total: int = Field(..., description="Toplam transaction sayısı")
    processed: int = Field(..., description="Başarıyla işlenen sayısı")
    failed: int = Field(..., description="Hata veren sayısı")
    results: List[BatchScoreItem]
    risk_summary: Dict[str, int] = Field(
        ..., description="risk_level bazında dağılım: {low, medium, high, very_high}"
    )


# ─── /explain ─────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    transaction: Dict[str, Any] = Field(..., description="Transaction fields")


class RetrievedDoc(BaseModel):
    id: str
    title: str
    category: str
    score: float = Field(..., description="Cosine similarity [0,1]")
    content: Optional[str] = None


class AuditEntry(BaseModel):
    """Adım 7 rule_audit_trail'den parse edilen tek kural kaydı."""
    rule_id: str
    triggered: bool
    multiplier: float
    severity: Optional[str] = None
    flag: Optional[str] = None
    explanation: Optional[str] = None


class ExplainResponse(BaseModel):
    transaction_id: Optional[Any] = None
    fraud_score: float
    narrative: str = Field(..., description="Deterministik transaction özeti")
    driver_features: List[str] = Field(..., description="Güçlü fraud sinyalleri")
    retrieved_docs: List[RetrievedDoc] = Field(..., description="RAG'den gelen KB dökümanları")
    explanation: str = Field(..., description="LLM açıklaması")
    strategy: Optional[str] = Field(None, description="Orchestrator stratejisi")
    dominant_signals: List[str] = Field(default_factory=list)
    # Adım 7 audit trail
    rule_audit_trail: List[AuditEntry] = Field(
        default_factory=list,
        description="Adım 7 rule_audit_trail — her kuralın tam değerlendirme kaydı"
    )


# ─── /rules/evaluate ──────────────────────────────────────────────────────────

class RulesEvaluateRequest(BaseModel):
    transaction: Dict[str, Any] = Field(..., description="Transaction fields")


class RuleDetail(BaseModel):
    rule_id: str
    severity: str
    multiplier: float
    explanation: str


class RulesEvaluateResponse(BaseModel):
    transaction_id: Optional[Any] = None
    rules_triggered: List[str]
    max_multiplier: float
    rule_score: float
    rule_flags: List[str]
    severity: str = Field(..., description="En yüksek tetiklenen severity")
    rule_details: List[RuleDetail]
    # Adım 7 tam audit trail
    rule_audit_trail: List[AuditEntry] = Field(
        default_factory=list,
        description="Tüm kuralların değerlendirme kaydı (tetiklenen + tetiklenmeyen)"
    )


# ─── /rag/query ───────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Serbest metin sorgusu veya transaction narrative")
    top_k: int = Field(default=3, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    query: str
    retrieved_docs: List[RetrievedDoc]
    answer: str = Field(..., description="LLM yanıtı")


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    faiss_loaded: bool
    kb_doc_count: int