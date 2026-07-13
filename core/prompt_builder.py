from __future__ import annotations

from typing import List, Dict, Any, Optional


# =========================================================
# TABLE FORMATTER (STRUCTURED OUTPUT)
# =========================================================

def format_structured_facts(rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    """
    Converts SQL result rows into a clean markdown table.
    This is REQUIRED for structured agent visibility.
    """

    if not rows:
        return "## Structured Facts\nNo rows returned."

    rows = rows[:max_rows]

    headers = list(rows[0].keys())

    table = "## Structured Facts\n"
    table += "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for r in rows:
        table += "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n"

    return table


# =========================================================
# SEMANTIC FORMATTER (QDRANT / VECTOR OUTPUT)
# =========================================================

def format_semantic_chunks(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
    """
    Formats vector search results (reviews, complaints, etc.)
    """

    if not chunks:
        return "## Key Evidence Snippets\nNo relevant context found."

    chunks = chunks[:max_chunks]

    block = "## Key Evidence Snippets\n"

    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "")[:300]
        score = c.get("score", 0.0)
        block += f"{i}. ★{round(score, 4)} — {text}\n"

    return block


# =========================================================
# MAIN PROMPT BUILDER
# =========================================================

class PromptBuilder:
    """
    Single unified prompt builder for structured + semantic + hybrid queries.
    """

    def __init__(self, max_context_chars: int = 6000):
        self.max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        structured_rows: Optional[List[Dict[str, Any]]] = None,
        semantic_chunks: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:

        structured_rows = structured_rows or []
        semantic_chunks = semantic_chunks or []
        metadata = metadata or {}

        parts: List[str] = []

        # =================================================
        # SYSTEM INSTRUCTION
        # =================================================

        parts.append(
            system_prompt or
            "You are a product analytics assistant. "
            "Answer ONLY using the provided structured and semantic context."
        )

        # =================================================
        # QUERY
        # =================================================

        parts.append(f"\n## Question\n{query}\n")

        # =================================================
        # STRUCTURED DATA (SQL)
        # =================================================

        if structured_rows:
            parts.append(
                format_structured_facts(structured_rows)
            )

        # =================================================
        # SEMANTIC DATA (QDRANT)
        # =================================================

        if semantic_chunks:
            parts.append(
                format_semantic_chunks(semantic_chunks)
            )

        # =================================================
        # METADATA (OPTIONAL DEBUG)
        # =================================================

        if metadata:
            parts.append("\n## Metadata")
            for k, v in metadata.items():
                parts.append(f"- {k}: {v}")

        # =================================================
        # FINAL MERGE
        # =================================================

        context = "\n\n".join(parts)

        # =================================================
        # SMART TRUNCATION (IMPORTANT FIX)
        # =================================================

        if len(context) > self.max_context_chars:
            context = self._smart_truncate(context)

        return context

    # =====================================================
    # SAFE TRUNCATION (NEVER BREAK TABLES)
    # =====================================================

    def _smart_truncate(self, text: str) -> str:
        """
        Prevents cutting structured tables mid-way.
        """

        if "## Structured Facts" in text:
            head, tail = text.split("## Structured Facts", 1)

            head = head[:3000]
            tail = tail[:self.max_context_chars]

            return head + "\n\n## Structured Facts" + tail

        return text[:self.max_context_chars] + "\n... (truncated)"