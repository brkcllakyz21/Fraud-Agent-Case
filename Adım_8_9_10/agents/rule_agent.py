from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base_agent import AgentMessage, BaseAgent


class RuleAgent(BaseAgent):

    def __init__(self, rules: List[Dict]) -> None:
        super().__init__("rule_agent")
        self.rules = rules

    def execute(self, message: AgentMessage) -> AgentMessage:
        try:
            tx = message.payload.get("transaction", {})
            rules = message.payload.get("rules", self.rules)

            triggered_rules = []
            rule_details = []
            multipliers = [1.0]
            flags = []
            severities = []

            for rule in rules:
                if not rule.get("enabled", True):
                    continue
                triggered, multiplier, explanation = _evaluate_rule(rule, tx)
                if triggered:
                    triggered_rules.append(rule["id"])
                    multipliers.append(multiplier)
                    flags.append(rule.get("flag", rule["id"]))
                    severities.append(rule.get("severity", "MEDIUM"))
                    rule_details.append({
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "MEDIUM"),
                        "multiplier": multiplier,
                        "explanation": explanation,
                    })

            max_multiplier = max(multipliers)
            n_total = len([r for r in rules if r.get("enabled", True)])
            rule_score = len(triggered_rules) / n_total if n_total > 0 else 0.0

            # En yüksek severity
            severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            top_severity = max(severities, key=lambda s: severity_order.get(s, 0)) if severities else "NONE"

            result = {
                "transaction_id": tx.get("TransactionID"),
                "rules_triggered": triggered_rules,
                "max_multiplier": max_multiplier,
                "rule_score": rule_score,
                "rule_flags": flags,
                "severity": top_severity,
                "rule_details": rule_details,
            }
            return self._success(message, result)
        except Exception as e:
            return self._failure(message, str(e))


def _get_val(tx: Dict, key: str) -> Optional[float]:
    val = tx.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _evaluate_condition(check: Dict, tx: Dict) -> bool:
    col = check.get("column")
    op = check.get("operator")
    threshold = check.get("value")
    val = _get_val(tx, col)
    if val is None:
        return False
    ops = {
        ">": val > threshold,
        ">=": val >= threshold,
        "<": val < threshold,
        "<=": val <= threshold,
        "==": val == threshold,
        "!=": val != threshold,
    }
    return ops.get(op, False)


def _evaluate_rule(rule: Dict, tx: Dict) -> tuple:
    logic = rule.get("logic", "AND")
    conditions = rule.get("conditions", [])
    multiplier = float(rule.get("multiplier", 1.0))
    explanation = rule.get("description", rule["id"])

    if not conditions:
        return False, multiplier, explanation

    results = [_evaluate_condition(c, tx) for c in conditions]
    triggered = all(results) if logic == "AND" else any(results)
    return triggered, multiplier, explanation
