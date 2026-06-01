from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def build_faiss_index(embeddings: np.ndarray):
    import faiss
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index, metadata: List[Dict], index_path: str) -> None:
    import faiss
    base = Path(index_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(base) + ".index")
    with open(str(base) + ".metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def load_index(index_path: str) -> Tuple:
    import faiss
    base = Path(index_path)
    index = faiss.read_index(str(base) + ".index")
    with open(str(base) + ".metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search(
    query_embedding: np.ndarray,
    index,
    metadata: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    import faiss
    q = query_embedding.copy()
    faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = dict(metadata[idx])
        doc["score"] = float(score)
        results.append(doc)
    return results
