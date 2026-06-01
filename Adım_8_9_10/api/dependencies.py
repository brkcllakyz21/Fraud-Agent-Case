from __future__ import annotations

from fastapi import Request


def get_orchestrator(request: Request):
    return request.app.state.container.orchestrator()


def get_explanation_agent(request: Request):
    return request.app.state.container.explanation_agent()


def get_rule_agent(request: Request):
    return request.app.state.container.rule_agent()


def get_pipeline_service(request: Request):
    return request.app.state.container.pipeline_service()


def get_rag_service(request: Request):
    container = request.app.state.container
    index, metadata = container.faiss_index(), container.faiss_metadata()
    return {
        "index": index,
        "metadata": metadata,
        "kb_docs": container.kb_docs(),
        "embedding_model": container.embedding_model(),
        "llm_base_url": request.app.state.llm_base_url,
        "llm_model": request.app.state.llm_model,
    }