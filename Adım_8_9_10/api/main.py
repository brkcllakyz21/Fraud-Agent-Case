from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from api.routers import explain, rag, rules, score
from api.schemas import HealthResponse
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Fraud Detection Platform starting up...")

    from containers import Container
    from dependency_injector import providers as _providers

    container = Container()
    app.state.container = container
    app.state.llm_base_url = settings.LM_STUDIO_BASE_URL
    app.state.llm_model = settings.LM_STUDIO_CHAT_MODEL

    # KB yükle
    logger.info("Loading knowledge base from %s", settings.KB_PATH)
    kb_docs = container.kb_docs()
    logger.info("Loaded %d KB documents.", len(kb_docs))

    # Embedding modeli yükle
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
    embedding_model = container.embedding_model()

    # FAISS: varsa yükle, yoksa oluştur ve kaydet
    faiss_index_file = settings.FAISS_INDEX_PATH + ".index"
    if not Path(faiss_index_file).exists():
        logger.info("FAISS index not found — building and saving...")
        from rag.embedder import embed_documents
        from rag.kb_builder import get_document_metadata, get_documents_for_embedding
        from rag.vector_store import build_faiss_index, save_index
        texts = get_documents_for_embedding(kb_docs)
        embeddings = embed_documents(texts, embedding_model)
        index = build_faiss_index(embeddings)
        metadata = get_document_metadata(kb_docs)
        save_index(index, metadata, settings.FAISS_INDEX_PATH)
        # Singleton'ları override et — diskten tekrar okumaya gerek yok
        container.faiss_index.override(_providers.Object(index))
        container.faiss_metadata.override(_providers.Object(metadata))
        logger.info("FAISS index built with %d documents.", len(kb_docs))
    else:
        logger.info("Loading existing FAISS index from %s", settings.FAISS_INDEX_PATH)
        container.faiss_index()
        container.faiss_metadata()
        logger.info("FAISS index loaded.")

    # LLM health check
    from rag.llm_client import check_llm_health
    llm_ok = check_llm_health(settings.LM_STUDIO_BASE_URL)
    if llm_ok:
        logger.info("LM Studio reachable at %s", settings.LM_STUDIO_BASE_URL)
    else:
        logger.warning(
            "LM Studio NOT reachable at %s — /explain and /rag/query will return fallback messages.",
            settings.LM_STUDIO_BASE_URL,
        )

    logger.info("Startup complete. Platform ready.")
    yield
    logger.info("Fraud Detection Platform shutting down.")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="IEEE-CIS Fraud Detection — Anomaly Scoring, Explainability, Rule Engine, RAG",
    lifespan=lifespan,
)

app.include_router(score.router)
app.include_router(explain.router)
app.include_router(rules.router)
app.include_router(rag.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    from rag.llm_client import check_llm_health
    llm_ok = check_llm_health(settings.LM_STUDIO_BASE_URL)
    try:
        container = app.state.container
        faiss_loaded = container.faiss_index() is not None
        kb_count = len(container.kb_docs())
    except Exception:
        faiss_loaded = False
        kb_count = 0
    return HealthResponse(
        status="ok",
        llm_available=llm_ok,
        faiss_loaded=faiss_loaded,
        kb_doc_count=kb_count,
    )