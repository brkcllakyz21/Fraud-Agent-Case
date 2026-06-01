from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from agents.base_agent import AgentMessage
from api.dependencies import get_orchestrator, get_pipeline_service
from api.schemas import AuditEntry, ExplainRequest, ExplainResponse, RetrievedDoc

router = APIRouter(prefix="/explain", tags=["explainability"])


def _parse_audit_trail(tx: Dict[str, Any]) -> List[AuditEntry]:
    raw = tx.get("rule_audit_trail", "")
    if not raw:
        return []
    try:
        entries = json.loads(raw) if isinstance(raw, str) else raw
        return [
            AuditEntry(
                rule_id=e.get("rule_id", e.get("id", "?")),
                triggered=e.get("triggered", False),
                multiplier=float(e.get("multiplier", 1.0)),
                severity=e.get("severity"),
                flag=e.get("flag") or e.get("rule_flag"),
                explanation=e.get("explanation"),
            )
            for e in entries if isinstance(e, dict)
        ]
    except (json.JSONDecodeError, TypeError):
        return []


@router.post("", response_model=ExplainResponse)
async def explain_transaction(
    request: ExplainRequest,
    pipeline_service=Depends(get_pipeline_service),
    orchestrator=Depends(get_orchestrator),
) -> ExplainResponse:
    """
    RAG pipeline + LLM kullanarak transaction'ın neden riskli olduğunu açıklar.
    Orchestrator üzerinden geçer — strateji ve dominant_signals dolu gelir.
    """
    tx = request.transaction

    # ── 1. Pipeline service — skor ve zenginleştirilmiş veri ─────────────
    pipeline_result = pipeline_service.score(tx)
    pipeline_dict   = pipeline_result.to_dict()
    enriched_tx     = dict(tx)
    for k, v in pipeline_dict.items():
        if k == "source":
            continue
        if v is None or v == 0.0:
            if enriched_tx.get(k) not in (None, 0.0):
                continue
        enriched_tx[k] = v

    # ── 2. Orchestrator — strateji belirle + explanation agent tetikle ────
    msg = AgentMessage(
        task="score",
        payload={"transaction": enriched_tx, "mode": "full"},  # full → her zaman explain
        sender="api",
        receiver="orchestrator",
    )
    result_msg = orchestrator.execute(msg)

    if not result_msg.success:
        raise HTTPException(status_code=500, detail=result_msg.error)

    r = result_msg.result

    retrieved = [
        RetrievedDoc(
            id=d.get("id", ""),
            title=d.get("title", ""),
            category=d.get("category", ""),
            score=d.get("score", 0.0),
            content=d.get("content"),
        )
        for d in r.get("retrieved_docs", [])
    ]

    audit_trail = _parse_audit_trail(enriched_tx)

    return ExplainResponse(
        transaction_id=r.get("transaction_id"),
        fraud_score=r.get("fraud_score", 0.0),
        narrative=r.get("narrative", ""),
        driver_features=r.get("driver_features", []),
        retrieved_docs=retrieved,
        explanation=r.get("explanation", ""),
        strategy=r.get("strategy"),
        dominant_signals=r.get("dominant_signals", []),
        rule_audit_trail=audit_trail,
    )