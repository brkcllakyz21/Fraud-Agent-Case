from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import AgentMessage, BaseAgent


class ScoringAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__("scoring_agent")

    def execute(self, message: AgentMessage) -> AgentMessage:
        try:
            tx = message.payload.get("transaction", {})

            rule_score = tx.get("rule_adjusted_score")
            context_score = tx.get("context_adjusted_score")
            fraud_score_raw = tx.get("fraud_score")

            # En güncel skoru al
            final_score = rule_score if rule_score is not None else (
                context_score if context_score is not None else (
                    fraud_score_raw if fraud_score_raw is not None else 0.0
                )
            )
            final_score = float(final_score)

            breakdown = {
                "column_score_mean": _safe_float(tx.get("column_score_mean")),
                "multivariate_score": _safe_float(tx.get("multivariate_score")),
                "entity_score": _safe_float(tx.get("entity_score")),
                "temporal_score": _safe_float(tx.get("temporal_score")),
                "context_adjusted_score": _safe_float(context_score),
                "rule_adjusted_score": _safe_float(rule_score),
            }

            result = {
                "transaction_id": tx.get("TransactionID"),
                "fraud_score": final_score,
                "risk_level": self._score_to_risk_level(final_score),
                "score_breakdown": breakdown,
            }
            return self._success(message, result)
        except Exception as e:
            return self._failure(message, str(e))

    @staticmethod
    def _score_to_risk_level(score: float) -> str:
        if score < 0.3:
            return "low"
        elif score < 0.5:
            return "medium"
        elif score < 0.7:
            return "high"
        else:
            return "very_high"


def _safe_float(val: Any) -> Any:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
