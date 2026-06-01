from __future__ import annotations

from typing import Dict, List


SYSTEM_PROMPT = """You are a fraud detection analyst. Explain why this transaction is flagged.
Use ONLY the provided signals and policies. Be direct and concise.
IMPORTANT: Respond ONLY in English. Never use Turkish or any other language.

Respond in this exact format (no thinking, no preamble):

**Summary:** [1-2 sentences]
**Key Risk Signals:** [bullet list of top 3 signals]
**Relevant Policies:** [which KB policies apply and why]
**Recommendation:** [block / review / monitor]"""


def build_rag_prompt(
    narrative: str,
    retrieved_docs: List[Dict],
    driver_features: List[str],
) -> List[Dict[str, str]]:
    drivers_text = "\n".join(f"- {f}" for f in driver_features) if driver_features else "- none detected"

    docs_text = ""
    for doc in retrieved_docs:
        docs_text += f"[{doc['id']}] {doc['title']}\n{doc.get('content', '')}\n\n"

    user_content = (
        f"--- TRANSACTION SIGNALS ---\n{narrative}\n\n"
        f"--- DETECTED RISK FEATURES ---\n{drivers_text}\n\n"
        f"--- RELEVANT FRAUD POLICIES (retrieved) ---\n{docs_text}"
        f"--- TASK ---\n"
        f"Explain why this transaction is flagged and provide a recommendation."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]