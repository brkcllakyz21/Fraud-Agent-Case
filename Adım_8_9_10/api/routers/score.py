from __future__ import annotations

import json
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from agents.base_agent import AgentMessage
from api.dependencies import get_orchestrator, get_pipeline_service
from api.schemas import (
    BatchScoreItem,
    BatchScoreRequest,
    BatchScoreResponse,
    ScoreBreakdown,
    ScoreRequest,
    ScoreResponse,
)

router = APIRouter(prefix="/score", tags=["scoring"])


def _build_score_response(r: Dict, breakdown_src: Dict) -> ScoreResponse:
    """Orchestrator sonucundan ScoreResponse oluşturur."""
    breakdown_data = r.get("score_breakdown") or breakdown_src
    return ScoreResponse(
        transaction_id=r.get("transaction_id"),
        fraud_score=r.get("fraud_score", 0.0),
        risk_level=r.get("risk_level", "unknown"),
        score_breakdown=ScoreBreakdown(**{
            k: v for k, v in breakdown_data.items()
            if k in ScoreBreakdown.model_fields
        }) if breakdown_data else None,
        strategy=r.get("strategy"),
        dominant_signals=r.get("dominant_signals", []),
        agents_called=r.get("agents_called", []),
        agent_errors=r.get("agent_errors", {}),
        pipeline_source=r.get("pipeline_source"),
    )


@router.post("", response_model=ScoreResponse)
async def score_transaction(
    request: ScoreRequest,
    pipeline_service=Depends(get_pipeline_service),
    orchestrator=Depends(get_orchestrator),
) -> ScoreResponse:
    """
    Hibrit scoring endpoint:
    1. TransactionID varsa cache'de arar (Adım 7 parquet)
    2. Cache hit → skorları doğrudan alır, orchestrator ile mode/strateji işler
    3. Cache miss → Adım 3-7 pipeline tetiklenir, yeni skor üretilir
    4. Pipeline da yoksa → gelen transaction alanlarını direkt kullanır
    """
    tx = request.transaction
    tx_id = tx.get("TransactionID")

    # ── 1. Pipeline service ile skor al (cache veya pipeline) ────────────
    pipeline_result = pipeline_service.score(tx)

    # Pipeline sonucunu transaction'a merge et
    # Ama pipeline 0.0 döndürdüyse (hata/eksik step) ham transaction'daki
    # skorları koru — elle verilen değerler ezilmesin
    enriched_tx = dict(tx)
    pipeline_dict = pipeline_result.to_dict()

    for k, v in pipeline_dict.items():
        if k == "source":
            continue
        # Pipeline 0.0 veya None döndürdüyse ham değeri koru
        if v is None or v == 0.0:
            if enriched_tx.get(k) not in (None, 0.0):
                continue  # ham değer daha iyi, atla
        enriched_tx[k] = v

    # ── 2. Orchestrator ile strateji + agent kararları ────────────────────
    msg = AgentMessage(
        task="score",
        payload={"transaction": enriched_tx, "mode": request.mode},
        sender="api",
        receiver="orchestrator",
    )
    result_msg = orchestrator.execute(msg)

    if not result_msg.success:
        raise HTTPException(status_code=500, detail=result_msg.error)

    r = result_msg.result

    # Pipeline kaynağını sonuca ekle
    r.setdefault("agent_errors", {})
    r["pipeline_source"] = pipeline_result.source

    return _build_score_response(r, pipeline_result.to_dict())


@router.post("/batch", response_model=BatchScoreResponse)
async def score_batch(
    request: BatchScoreRequest,
    pipeline_service=Depends(get_pipeline_service),
    orchestrator=Depends(get_orchestrator),
) -> BatchScoreResponse:
    """
    Toplu transaction scoring. Her transaction için hibrit strateji çalışır.
    Max 1000 transaction per request.
    """
    results: List[BatchScoreItem] = []
    risk_summary: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
    failed = 0

    for tx in request.transactions:
        try:
            # Pipeline service
            pipeline_result = pipeline_service.score(tx)
            # Pipeline sonucu boşsa ham transaction skorlarını koru
            pipeline_dict = pipeline_result.to_dict()
            enriched_tx = dict(tx)
            for k, v in pipeline_dict.items():
                if k == "source":
                    continue
                if v is None or v == 0.0:
                    if enriched_tx.get(k) not in (None, 0.0):
                        continue
                enriched_tx[k] = v

            msg = AgentMessage(
                task="score",
                payload={"transaction": enriched_tx, "mode": request.mode},
                sender="api",
                receiver="orchestrator",
            )
            result_msg = orchestrator.execute(msg)

            if result_msg.success:
                r = result_msg.result
                risk_level = r.get("risk_level", "unknown")
                if risk_level in risk_summary:
                    risk_summary[risk_level] += 1
                results.append(BatchScoreItem(
                    transaction_id=r.get("transaction_id"),
                    fraud_score=r.get("fraud_score", 0.0),
                    risk_level=risk_level,
                    strategy=r.get("strategy"),
                    agent_errors=r.get("agent_errors", {}),
                ))
            else:
                failed += 1
                results.append(BatchScoreItem(
                    transaction_id=tx.get("TransactionID"),
                    fraud_score=0.0,
                    risk_level="unknown",
                    agent_errors={"orchestrator": result_msg.error or "unknown"},
                ))
        except Exception as e:
            failed += 1
            results.append(BatchScoreItem(
                transaction_id=tx.get("TransactionID"),
                fraud_score=0.0,
                risk_level="unknown",
                agent_errors={"pipeline": str(e)},
            ))

    return BatchScoreResponse(
        total=len(request.transactions),
        processed=len(request.transactions) - failed,
        failed=failed,
        results=results,
        risk_summary=risk_summary,
    )