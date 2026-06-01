from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import AgentMessage, BaseAgent
from rag.rag_pipeline import run_rag_pipeline


# Strateji → hangi KB kategorileri öncelikli aransın
STRATEGY_KB_HINTS = {
    "statistical":             ["model_signal", "feature_combination", "amount_anomaly"],
    "statistical_new_card":    ["model_signal", "feature_combination", "velocity_fraud"],
    "behavioral":              ["model_signal", "trusted_entity", "identity_mismatch"],
    "behavioral_multivariate": ["model_signal", "composite_risk", "identity_mismatch"],
    "multivariate":            ["model_signal", "composite_risk", "feature_combination"],
    "temporal":                ["temporal_velocity", "nighttime_fraud", "model_signal"],
    "rule_based":              ["general_policy", "composite_risk"],
}


class ExplanationAgent(BaseAgent):

    def __init__(
        self,
        index,
        metadata: List[Dict],
        kb_docs: List[Dict],
        embedding_model,
        llm_base_url: str,
        llm_model: str,
        top_k: int = 3,
        llm_timeout: int = 180,
    ) -> None:
        super().__init__("explanation_agent")
        self.index = index
        self.metadata = metadata
        self.kb_docs = kb_docs
        self.embedding_model = embedding_model
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.top_k = top_k
        self.llm_timeout = llm_timeout

    def execute(self, message: AgentMessage) -> AgentMessage:
        try:
            tx = message.payload.get("transaction", {})

            # Orchestrator'dan gelen strateji bilgisi
            strategy = tx.pop("_strategy", "unknown")
            dominant_signals = tx.pop("_dominant_signals", [])
            signal_pattern = tx.pop("_signal_pattern", "unknown")

            # Strateji context'ini narrative query'ye ekle
            enriched_tx = self._enrich_with_strategy(tx, strategy, dominant_signals, signal_pattern)

            result = run_rag_pipeline(
                row=enriched_tx,
                index=self.index,
                metadata=self.metadata,
                kb_docs=self.kb_docs,
                embedding_model=self.embedding_model,
                llm_base_url=self.llm_base_url,
                llm_model=self.llm_model,
                top_k=self.top_k,
                llm_timeout=self.llm_timeout,
            )

            # Strateji bilgisini sonuca ekle
            result["strategy"] = strategy
            result["signal_pattern"] = signal_pattern
            result["dominant_signals"] = dominant_signals

            return self._success(message, result)
        except Exception as e:
            return self._failure(message, str(e))

    def _enrich_with_strategy(
        self,
        tx: Dict[str, Any],
        strategy: str,
        dominant_signals: List[str],
        signal_pattern: str,
    ) -> Dict[str, Any]:
        """
        Strateji bilgisini transaction dict'ine ekler.
        narrative.py bunu okuyarak daha zengin açıklama üretir.
        """
        enriched = dict(tx)

        # KB hint kategorilerini ekle — RAG bu kategorilerdeki dökümanları önceliklendirir
        kb_hints = STRATEGY_KB_HINTS.get(strategy, [])
        enriched["_kb_category_hints"] = kb_hints
        enriched["_signal_pattern"] = signal_pattern
        enriched["_dominant_signals_text"] = "; ".join(dominant_signals) if dominant_signals else ""

        # Strateji açıklamasını narrative'e girecek alana ekle
        strategy_descriptions = {
            "statistical":
                "Primary anomaly: statistical feature deviation (column scores elevated).",
            "statistical_new_card":
                "Primary anomaly: statistical deviation with low entity history — possible new/stolen card.",
            "behavioral":
                "Primary anomaly: entity behavioral deviation from historical baseline.",
            "behavioral_multivariate":
                "Primary anomaly: entity behavioral deviation confirmed by multivariate analysis — account takeover pattern.",
            "multivariate":
                "Primary anomaly: joint feature distribution anomaly detected by Isolation Forest.",
            "temporal":
                "Primary anomaly: temporal pattern deviation from entity's normal activity hours.",
            "rule_based":
                "Primary signal: business rule triggers — model anomaly signals are weak.",
        }
        enriched["_strategy_description"] = strategy_descriptions.get(strategy, "")

        return enriched