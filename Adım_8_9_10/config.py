from __future__ import annotations

import os


class Settings:
    # LM Studio
    LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_CHAT_MODEL: str = os.getenv("LM_STUDIO_CHAT_MODEL", "google/gemma-4-e4b")
    LM_STUDIO_TIMEOUT: int = int(os.getenv("LM_STUDIO_TIMEOUT", "180"))

    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # FAISS
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "knowledge_base/faiss_index")
    FAISS_TOP_K: int = int(os.getenv("FAISS_TOP_K", "3"))

    # Knowledge Base
    KB_PATH: str = os.getenv("KB_PATH", "knowledge_base/fraud_kb.json")

    # Scoring pipeline outputs (Adım 5-7 çıktıları)
    STEP7_OUTPUT_PATH: str = os.getenv("STEP7_OUTPUT_PATH", "../Adım 7/outputs/step7/df_rules_train.parquet")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_TITLE: str = "Fraud Detection Platform"
    API_VERSION: str = "1.0.0"

    # LLM generation
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))


settings = Settings()