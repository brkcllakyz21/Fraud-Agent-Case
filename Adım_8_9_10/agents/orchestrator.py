from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import AgentMessage, BaseAgent
from agents.explanation_agent import ExplanationAgent
from agents.rule_agent import RuleAgent
from agents.scoring_agent import ScoringAgent

logger = logging.getLogger(__name__)

# Adım 5 AUC ağırlıkları — eşik kararlarında referans
SIGNAL_WEIGHTS = {
    "column_score_mean":  {"auc": 0.7309, "weight": 0.3709},
    "column_score_max":   {"auc": 0.7024, "weight": 0.3251},
    "multivariate_score": {"auc": 0.6413, "weight": 0.2270},
    "entity_score":       {"auc": 0.5302, "weight": 0.0485},
    "temporal_score":     {"auc": 0.5178, "weight": 0.0286},
}

# Strateji eşikleri
THRESHOLDS = {
    "column_mean_high":    0.60,   # column_score_mean > bu → statistical strateji
    "column_max_high":     0.55,   # column_score_max > bu → statistical strateji
    "multivariate_high":   0.55,   # multivariate_score > bu → multivariate strateji
    "entity_high":         0.50,   # entity_score > bu → behavioral strateji
    "temporal_high":       0.50,   # temporal_score > bu → temporal strateji
    "entity_low_col_high": 0.30,   # entity < bu + column > 0.6 → new_card pattern
    "score_explain":       0.50,   # fraud_score > bu → ExplanationAgent çalışır
    "score_rules":         0.30,   # fraud_score > bu → RuleAgent çalışır
}


def _detect_strategy(tx: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Transaction sinyallerine göre hangi katmanın baskın olduğunu belirler.

    Returns
    -------
    Tuple[str, List[str], Dict[str, Any]]
        (strategy_name, dominant_signals, context_for_explanation)
    """
    col_mean  = _f(tx.get("column_score_mean"))
    col_max   = _f(tx.get("column_score_max"))
    mv_score  = _f(tx.get("multivariate_score"))
    ent_score = _f(tx.get("entity_score"))
    tmp_score = _f(tx.get("temporal_score"))

    dominant_signals: List[str] = []
    context: Dict[str, Any] = {
        "column_score_mean":  col_mean,
        "column_score_max":   col_max,
        "multivariate_score": mv_score,
        "entity_score":       ent_score,
        "temporal_score":     tmp_score,
    }

    # ── 1. Statistical strateji — en güçlü sinyal (AUC 0.73, ağırlık %37) ──
    if (col_mean is not None and col_mean > THRESHOLDS["column_mean_high"]) or \
       (col_max  is not None and col_max  > THRESHOLDS["column_max_high"]):
        if col_mean is not None and col_mean > THRESHOLDS["column_mean_high"]:
            dominant_signals.append(f"column_score_mean={col_mean:.3f} (AUC 0.73, weight 37%)")
        if col_max is not None and col_max > THRESHOLDS["column_max_high"]:
            dominant_signals.append(f"column_score_max={col_max:.3f} (AUC 0.70, weight 33%)")

        # New card pattern: entity düşük ama column yüksek
        if ent_score is not None and ent_score < THRESHOLDS["entity_low_col_high"]:
            dominant_signals.append(f"entity_score={ent_score:.3f} (low — possible new card fraud)")
            context["pattern"] = "new_card_fraud"
            return "statistical_new_card", dominant_signals, context

        context["pattern"] = "statistical_anomaly"
        return "statistical", dominant_signals, context

    # ── 2. Behavioral strateji — entity sapması ─────────────────────────────
    if ent_score is not None and ent_score > THRESHOLDS["entity_high"]:
        dominant_signals.append(f"entity_score={ent_score:.3f} (AUC 0.53, weight 5%)")
        if mv_score is not None and mv_score > THRESHOLDS["multivariate_high"]:
            dominant_signals.append(f"multivariate_score={mv_score:.3f} — confirms joint anomaly")
            context["pattern"] = "account_takeover"
            return "behavioral_multivariate", dominant_signals, context
        context["pattern"] = "behavioral_deviation"
        return "behavioral", dominant_signals, context

    # ── 3. Multivariate strateji — joint distribution anomalisi ─────────────
    if mv_score is not None and mv_score > THRESHOLDS["multivariate_high"]:
        dominant_signals.append(f"multivariate_score={mv_score:.3f} (AUC 0.64, weight 23%)")
        context["pattern"] = "joint_distribution_anomaly"
        return "multivariate", dominant_signals, context

    # ── 4. Temporal strateji — zaman anomalisi ──────────────────────────────
    if tmp_score is not None and tmp_score > THRESHOLDS["temporal_high"]:
        dominant_signals.append(f"temporal_score={tmp_score:.3f} (AUC 0.52, weight 3%)")
        context["pattern"] = "temporal_anomaly"
        return "temporal", dominant_signals, context

    # ── 5. Rule-based strateji — model sinyali yoksa kurallara bak ──────────
    context["pattern"] = "rule_based"
    return "rule_based", [], context


def _f(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class OrchestratorAgent(BaseAgent):
    """
    Sinyal bazlı dallanma yapan multi-agent orchestrator.

    Karar mantığı:
    - Hangi anomali katmanı baskın? → strateji belirle
    - Strateji → hangi agent'lar çalışır + ExplanationAgent'a ne context gider
    - Hata yalıtımı: bir agent başarısız olursa diğerleri çalışmaya devam eder

    Stratejiler:
    - statistical        : column_score yüksek → istatistiksel sapma açıklaması
    - statistical_new_card: column yüksek + entity düşük → yeni kart fraud pattern
    - behavioral         : entity_score yüksek → davranışsal sapma
    - behavioral_multivariate: entity + mv yüksek → hesap ele geçirme
    - multivariate       : mv_score yüksek → joint distribution anomali
    - temporal           : temporal_score yüksek → zaman anomalisi
    - rule_based         : model sinyali zayıf → kural bazlı karar
    """

    def __init__(
        self,
        scoring_agent: ScoringAgent,
        explanation_agent: ExplanationAgent,
        rule_agent: RuleAgent,
    ) -> None:
        super().__init__("orchestrator")
        self.scoring_agent = scoring_agent
        self.explanation_agent = explanation_agent
        self.rule_agent = rule_agent

    def execute(self, message: AgentMessage) -> AgentMessage:
        try:
            tx = message.payload.get("transaction", {})
            mode = message.payload.get("mode", "auto")

            agents_called: List[str] = []
            agent_errors: Dict[str, str] = {}
            result: Dict[str, Any] = {"transaction_id": tx.get("TransactionID")}

            # ── 1. ScoringAgent — her zaman çalışır ─────────────────────────
            score_msg = self.delegate(self.scoring_agent, "score", {"transaction": tx})
            agents_called.append("scoring_agent")
            if score_msg.success:
                result.update(score_msg.result)
            else:
                agent_errors["scoring_agent"] = score_msg.error
                result["fraud_score"] = 0.0
                result["risk_level"] = "unknown"

            fraud_score = result.get("fraud_score", 0.0)

            # ── 2. Strateji tespiti ──────────────────────────────────────────
            # auto, full, score_and_rules modlarında da sinyal bazlı strateji belirle
            if mode in ("auto", "full", "score_and_rules", "score_only"):
                strategy, dominant_signals, signal_context = _detect_strategy(tx)
            else:
                strategy = mode
                dominant_signals = []
                signal_context = {"pattern": mode}

            result["strategy"] = strategy
            result["dominant_signals"] = dominant_signals
            result["signal_context"] = signal_context

            self.logger.info(
                "Strategy: %s | dominant: %s | fraud_score: %.3f",
                strategy, dominant_signals, fraud_score,
            )

            # ── 3. RuleAgent — strateji veya skora göre ─────────────────────
            run_rules = (
                mode in ("score_and_rules", "full")
                or (mode == "auto" and (fraud_score >= THRESHOLDS["score_rules"] or strategy != "rule_based"))
            )
            if run_rules:
                rule_msg = self.delegate(self.rule_agent, "evaluate_rules", {"transaction": tx})
                agents_called.append("rule_agent")
                if rule_msg.success:
                    result["rules"] = rule_msg.result
                else:
                    agent_errors["rule_agent"] = rule_msg.error

            # ── 4. ExplanationAgent — strateji context'iyle ─────────────────
            run_explain = (
                mode == "full"
                or (mode == "auto" and (
                    fraud_score >= THRESHOLDS["score_explain"]
                    or strategy in ("statistical", "statistical_new_card", "behavioral_multivariate")
                ))
            )
            if run_explain:
                # Adım 4 driver kolonlarını + strateji context'ini explanation'a geç
                enriched_tx = dict(tx)
                enriched_tx["_strategy"] = strategy
                enriched_tx["_dominant_signals"] = dominant_signals
                enriched_tx["_signal_pattern"] = signal_context.get("pattern", "unknown")

                exp_msg = self.delegate(
                    self.explanation_agent,
                    "explain",
                    {"transaction": enriched_tx},
                )
                agents_called.append("explanation_agent")
                if exp_msg.success:
                    result["explanation"] = exp_msg.result.get("explanation", "")
                    result["narrative"] = exp_msg.result.get("narrative", "")
                    result["retrieved_docs"] = exp_msg.result.get("retrieved_docs", [])
                    result["driver_features"] = exp_msg.result.get("driver_features", [])
                else:
                    agent_errors["explanation_agent"] = exp_msg.error

            result["agents_called"] = agents_called
            result["agent_errors"] = agent_errors
            return self._success(message, result)

        except Exception as e:
            return self._failure(message, str(e))

    def delegate(self, agent: BaseAgent, task: str, payload: Dict[str, Any]) -> AgentMessage:
        msg = AgentMessage(
            task=task,
            payload=payload,
            sender=self.agent_id,
            receiver=agent.agent_id,
        )
        try:
            return agent.execute(msg)
        except Exception as e:
            msg.set_error(str(e))
            self.logger.error("Delegation to %s failed: %s", agent.agent_id, e)
            return msg