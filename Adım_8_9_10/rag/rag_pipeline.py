from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.embedder import embed_query
from rag.llm_client import query_llm
from rag.narrative import extract_driver_features, transaction_to_narrative
from rag.prompt_builder import build_rag_prompt
from rag.vector_store import search


def run_rag_pipeline(
    row: Dict[str, Any],
    index,
    metadata: List[Dict],
    kb_docs: List[Dict],
    embedding_model,
    llm_base_url: str,
    llm_model: str,
    top_k: int = 3,
    max_tokens: int = 512,
    temperature: float = 0.2,
    llm_timeout: int = 60,
) -> Dict[str, Any]:
    # 1. Narrative ve driver features
    narrative = transaction_to_narrative(row)
    drivers = extract_driver_features(row)

    # 2. Query embedding
    query_embedding = embed_query(narrative, embedding_model)

    # 3. Vector search
    retrieved = search(query_embedding, index, metadata, top_k=top_k)

    # 4. Content'i kb_docs'tan join et (metadata sadece id/title taşıyor olabilir)
    kb_by_id = {doc["id"]: doc for doc in kb_docs}
    for doc in retrieved:
        if "content" not in doc or not doc["content"]:
            kb_doc = kb_by_id.get(doc["id"], {})
            doc["content"] = kb_doc.get("content", "")

    # 5. Prompt oluştur
    messages = build_rag_prompt(narrative, retrieved, drivers)

    # 6. LLM çağrısı
    explanation = query_llm(
        messages=messages,
        base_url=llm_base_url,
        model=llm_model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=llm_timeout,
    )

    # 7. Sonuç
    fraud_score = row.get("rule_adjusted_score") or row.get("context_adjusted_score") or row.get("fraud_score", 0.0)
    transaction_id = row.get("TransactionID")

    return {
        "narrative": narrative,
        "driver_features": drivers,
        "retrieved_docs": retrieved,
        "explanation": explanation,
        "fraud_score": float(fraud_score) if fraud_score is not None else 0.0,
        "transaction_id": transaction_id,
    }
