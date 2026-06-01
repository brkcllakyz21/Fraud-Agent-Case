from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from agents.base_agent import AgentMessage
from api.dependencies import get_rule_agent
from api.schemas import AuditEntry, RuleDetail, RulesEvaluateRequest, RulesEvaluateResponse

router = APIRouter(prefix="/rules", tags=["rules"])


def _build_audit_trail(
    all_rules: List[Dict],
    triggered_ids: List[str],
    rule_details: List[Dict],
) -> List[AuditEntry]:
    """
    Tüm kurallar için audit trail oluşturur — tetiklenen + tetiklenmeyen.
    """
    detail_map = {d["rule_id"]: d for d in rule_details}
    entries = []
    for rule in all_rules:
        rid = rule.get("id", "?")
        triggered = rid in triggered_ids
        detail = detail_map.get(rid, {})
        entries.append(AuditEntry(
            rule_id=rid,
            triggered=triggered,
            multiplier=float(detail.get("multiplier", rule.get("multiplier", 1.0))),
            severity=detail.get("severity") or rule.get("severity"),
            flag=rule.get("flag"),
            explanation=detail.get("explanation") or rule.get("description", ""),
        ))
    return entries


@router.post("/evaluate", response_model=RulesEvaluateResponse)
async def evaluate_rules(
    request: RulesEvaluateRequest,
    rule_agent=Depends(get_rule_agent),
) -> RulesEvaluateResponse:
    """
    Bir transaction üzerinde kural motorunu çalıştırır.
    Tetiklenen + tetiklenmeyen tüm kuralların tam audit trail'ini döndürür.
    """
    msg = AgentMessage(
        task="evaluate_rules",
        payload={"transaction": request.transaction},
        sender="api",
        receiver="rule_agent",
    )
    result_msg = rule_agent.execute(msg)

    if not result_msg.success:
        raise HTTPException(status_code=500, detail=result_msg.error)

    r = result_msg.result
    details = [RuleDetail(**d) for d in r.get("rule_details", [])]

    # Tam audit trail — agent'ın elindeki tüm kurallardan üret
    audit_trail = _build_audit_trail(
        all_rules=rule_agent.rules,
        triggered_ids=r.get("rules_triggered", []),
        rule_details=r.get("rule_details", []),
    )

    return RulesEvaluateResponse(
        transaction_id=r.get("transaction_id"),
        rules_triggered=r.get("rules_triggered", []),
        max_multiplier=r.get("max_multiplier", 1.0),
        rule_score=r.get("rule_score", 0.0),
        rule_flags=r.get("rule_flags", []),
        severity=r.get("severity", "NONE"),
        rule_details=details,
        rule_audit_trail=audit_trail,
    )