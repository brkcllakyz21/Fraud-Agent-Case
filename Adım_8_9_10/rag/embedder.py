from __future__ import annotations

from typing import List, Optional

import numpy as np


def load_embedding_model(model_name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def embed_documents(texts: List[str], model) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def embed_query(text: str, model) -> np.ndarray:
    embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    return embedding.astype(np.float32)
