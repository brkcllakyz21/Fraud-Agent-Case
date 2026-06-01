from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_knowledge_base(kb_path: str) -> List[Dict]:
    path = Path(kb_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs


def get_documents_for_embedding(kb_docs: List[Dict]) -> List[str]:
    return [f"{doc['title']}. {doc['content']}" for doc in kb_docs]


def get_document_metadata(kb_docs: List[Dict]) -> List[Dict]:
    return [
        {
            "id": doc["id"],
            "category": doc["category"],
            "title": doc["title"],
            "tags": doc.get("tags", []),
            "content": doc["content"],
        }
        for doc in kb_docs
    ]
