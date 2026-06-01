from __future__ import annotations

from dependency_injector import containers, providers

from config import settings


def _load_kb():
    from rag.kb_builder import load_knowledge_base
    return load_knowledge_base(settings.KB_PATH)


def _load_embedding_model():
    from rag.embedder import load_embedding_model
    return load_embedding_model(settings.EMBEDDING_MODEL, settings.EMBEDDING_DEVICE)


def _load_faiss_index():
    from rag.vector_store import load_index
    index, _ = load_index(settings.FAISS_INDEX_PATH)
    return index


def _load_faiss_metadata():
    from rag.vector_store import load_index
    _, metadata = load_index(settings.FAISS_INDEX_PATH)
    return metadata


def _load_rules():
    import os
    import yaml
    rules_path = os.getenv("RULES_PATH", "rules/fraud_rules.yaml")
    if not os.path.exists(rules_path):
        return []
    with open(rules_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = []
    for r in data.get("rules", []):
        if not r.get("enabled", True):
            continue

        # Adım 7 formatı: conditions.checks + action altında multiplier/flag/severity
        raw_conditions = r.get("conditions", {})
        action = r.get("action", {})

        if isinstance(raw_conditions, dict):
            # Adım 7 formatı: {logic: AND, checks: [{field, op, value}, ...]}
            logic = raw_conditions.get("logic", "AND")
            checks = raw_conditions.get("checks", [])
            # field/op → column/operator normalizasyonu
            conditions = [
                {
                    "column": c.get("field") or c.get("column"),
                    "operator": c.get("op") or c.get("operator"),
                    "value": c.get("value"),
                }
                for c in checks
                if isinstance(c, dict)
            ]
            multiplier = action.get("multiplier", r.get("multiplier", 1.0))
            flag = action.get("rule_flag") or r.get("flag", r["id"])
            severity = action.get("severity") or r.get("severity", "MEDIUM")
            description = action.get("explanation") or r.get("description", r["id"])
        else:
            # Basit format: conditions direkt liste
            logic = r.get("logic", "AND")
            conditions = raw_conditions if isinstance(raw_conditions, list) else []
            multiplier = r.get("multiplier", 1.0)
            flag = r.get("flag", r["id"])
            severity = r.get("severity", "MEDIUM")
            description = r.get("description", r["id"])

        rules.append({
            "id": r["id"],
            "description": description,
            "severity": severity,
            "enabled": True,
            "logic": logic,
            "multiplier": float(multiplier),
            "flag": flag,
            "conditions": conditions,
        })
    return rules


def _make_scoring_agent():
    from agents.scoring_agent import ScoringAgent
    return ScoringAgent()


def _make_rule_agent():
    from agents.rule_agent import RuleAgent
    rules = _load_rules()
    return RuleAgent(rules=rules)


def _make_explanation_agent(kb_docs, faiss_index, faiss_metadata, embedding_model):
    from agents.explanation_agent import ExplanationAgent
    return ExplanationAgent(
        index=faiss_index,
        metadata=faiss_metadata,
        kb_docs=kb_docs,
        embedding_model=embedding_model,
        llm_base_url=settings.LM_STUDIO_BASE_URL,
        llm_model=settings.LM_STUDIO_CHAT_MODEL,
        top_k=settings.FAISS_TOP_K,
        llm_timeout=settings.LM_STUDIO_TIMEOUT,
    )


def _make_orchestrator(scoring_agent, explanation_agent, rule_agent):
    from agents.orchestrator import OrchestratorAgent
    return OrchestratorAgent(
        scoring_agent=scoring_agent,
        explanation_agent=explanation_agent,
        rule_agent=rule_agent,
    )




def _make_score_cache():
    from pipeline.pipeline_service import ScoreCache
    import logging
    cache = ScoreCache(settings.STEP7_OUTPUT_PATH)
    try:
        cache.load()
    except Exception as e:
        logging.getLogger(__name__).warning("ScoreCache load failed: %s", e)
    return cache


def _make_pipeline_runner():
    import os
    from pipeline.pipeline_service import PipelineRunner
    rules_path = os.getenv("RULES_PATH", "rules/fraud_rules.yaml")
    return PipelineRunner(rules_path=rules_path)


def _make_pipeline_service(cache, runner):
    from pipeline.pipeline_service import PipelineService
    return PipelineService(cache=cache, runner=runner)

class Container(containers.DeclarativeContainer):
    """
    dependency_injector Container — Fraud Detection Platform

    Tasarım desenleri:
    - Singleton     : Pahalı nesneler (FAISS, embedding model, agent'lar) bir kez oluşturulur.
    - Factory       : Hafif geçici nesneler her çağrıda yeniden üretilir.
    - Configuration : Tüm ayarlar config.settings'ten beslenir, test'te override edilebilir.
    - Strategy      : llm_model değiştirilerek farklı LLM'e geçiş yapılabilir.
    - Repository    : kb_docs, knowledge base için repository rolü üstlenir.
    """

    # ── Knowledge Base (Repository pattern) ────────────────────────────────
    kb_docs = providers.Singleton(_load_kb)

    # ── Embedding Model (Singleton — RAM'e bir kez yüklenir) ───────────────
    embedding_model = providers.Singleton(_load_embedding_model)

    # ── FAISS Index ve Metadata ayrı Singleton'lar ─────────────────────────
    faiss_index = providers.Singleton(_load_faiss_index)
    faiss_metadata = providers.Singleton(_load_faiss_metadata)

    # ── Agents (Singleton) ──────────────────────────────────────────────────
    scoring_agent = providers.Singleton(_make_scoring_agent)

    rule_agent = providers.Singleton(_make_rule_agent)

    explanation_agent = providers.Singleton(
        _make_explanation_agent,
        kb_docs=kb_docs,
        faiss_index=faiss_index,
        faiss_metadata=faiss_metadata,
        embedding_model=embedding_model,
    )

    orchestrator = providers.Singleton(
        _make_orchestrator,
        scoring_agent=scoring_agent,
        explanation_agent=explanation_agent,
        rule_agent=rule_agent,
    )

    # ── Pipeline Service (Score Cache + Fallback Pipeline) ──────────────
    score_cache = providers.Singleton(_make_score_cache)
    pipeline_runner = providers.Singleton(_make_pipeline_runner)
    pipeline_service = providers.Singleton(
        _make_pipeline_service,
        cache=score_cache,
        runner=pipeline_runner,
    )