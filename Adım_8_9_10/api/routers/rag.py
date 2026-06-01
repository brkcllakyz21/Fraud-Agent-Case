from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_rag_service
from api.schemas import RAGQueryRequest, RAGQueryResponse, RetrievedDoc
from rag.embedder import embed_query
from rag.llm_client import query_llm
from rag.prompt_builder import build_rag_prompt
from rag.vector_store import search

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    rag_service=Depends(get_rag_service),
) -> RAGQueryResponse:
    try:
        index = rag_service["index"]
        metadata = rag_service["metadata"]
        kb_docs = rag_service["kb_docs"]
        embedding_model = rag_service["embedding_model"]
        llm_base_url = rag_service["llm_base_url"]
        llm_model = rag_service["llm_model"]

        # Embed query
        query_embedding = embed_query(request.query, embedding_model)

        # Vector search
        retrieved = search(query_embedding, index, metadata, top_k=request.top_k)

        # Content join
        kb_by_id = {doc["id"]: doc for doc in kb_docs}
        for doc in retrieved:
            if not doc.get("content"):
                doc["content"] = kb_by_id.get(doc["id"], {}).get("content", "")

        # LLM call — serbest sorgu için basit prompt
        messages = build_rag_prompt(
            narrative=request.query,
            retrieved_docs=retrieved,
            driver_features=[],
        )
        answer = query_llm(
            messages=messages,
            base_url=llm_base_url,
            model=llm_model,
            timeout=360,  # /rag/query için sabit, config'den de alınabilir
        )

        retrieved_docs = [
            RetrievedDoc(
                id=d.get("id", ""),
                title=d.get("title", ""),
                category=d.get("category", ""),
                score=d.get("score", 0.0),
                content=d.get("content"),
            )
            for d in retrieved
        ]
        return RAGQueryResponse(
            query=request.query,
            retrieved_docs=retrieved_docs,
            answer=answer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))