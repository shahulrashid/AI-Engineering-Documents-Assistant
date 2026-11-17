"""Prompt templates for grounded, citation-based answers."""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are an engineering documentation assistant. Answer the user's question \
using ONLY the numbered context excerpts provided.

Rules:
- If the context does not contain the answer, reply exactly: "I could not find this in the uploaded documents." \
Do not guess or fill gaps from general knowledge.
- Cite the excerpt number for every factual statement, for example [1] or [2][3].
- Preserve units, part numbers, torque values, and tolerances exactly as written.
- For procedures, present the steps as a numbered list in the order given in the documents.
- If excerpts conflict, say so and cite both.
- Do not add safety-critical instructions that are not in the documents. Recommend the official manual for anything unclear.
"""


def _format_history(history: list[dict[str, Any]] | None, max_turns: int = 6) -> str:
    if not history:
        return "(none)"
    recent = history[-max_turns:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def build_user_prompt(question: str, sources: list[Any], history: list[dict[str, Any]] | None = None) -> str:
    context = "\n\n".join(
        f"[{i}] Document: {s.doc_name}, page {s.page}\n{s.text}" for i, s in enumerate(sources, start=1)
    )
    return (
        f"Context excerpts:\n{context}\n\n"
        f"Conversation so far:\n{_format_history(history)}\n\n"
        f"Question: {question}"
    )
