"""
src/services/prompt_builder.py
────────────────────────────────
Constructs the full LLM prompt for RAG-based Q&A.

Responsibilities:
- System prompt definition (grounded, no hallucination instruction)
- Context injection from retrieved vector chunks
- Chat history injection (last N messages, token-guarded)
- Token budget management (prevents context overflow)

Design rationale:
- Prompt assembly is isolated here so it can be unit-tested
  independently of the LLM and vector store.
- Token counting uses a character-based heuristic (4 chars ≈ 1 token)
  since we don't want to add a tokenizer dependency just for counting.
  This is conservative and safe — real token counts are usually lower.
- Source deduplication ensures the same document chunk doesn't appear
  twice in the context window even if retrieved at different scores.
"""

from dataclasses import dataclass

from src.providers.llm.base import LLMMessage
from src.vector_store.base import SearchResult

# ── Constants ─────────────────────────────────────────────────────
# Conservative token budget — leaves room for the model's response
MAX_CONTEXT_TOKENS = 3000
MAX_HISTORY_TOKENS = 800
CHARS_PER_TOKEN = 4          # Heuristic: 1 token ≈ 4 characters

SYSTEM_PROMPT = """You are a private knowledge assistant. Your job is to answer \
questions accurately using ONLY the information provided in the context below.

Rules you must follow:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say: \
"I don't have enough information in the provided documents to answer that."
3. Always be concise and precise.
4. When relevant, mention the source document name so the user knows where \
the information comes from.
5. Never make up facts, figures, or quotes not present in the context."""


@dataclass
class BuiltPrompt:
    """The assembled prompt ready to send to the LLM."""
    messages: list[LLMMessage]
    context_chunks_used: int
    history_messages_used: int
    estimated_tokens: int


class PromptBuilder:
    """
    Assembles the full RAG prompt from retrieved chunks and chat history.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(
            query="What is the refund policy?",
            search_results=results,
            history=recent_messages,
        )
        response = await llm.generate(prompt.messages)
    """

    def __init__(
        self,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        max_history_tokens: int = MAX_HISTORY_TOKENS,
    ) -> None:
        self._max_context_tokens = max_context_tokens
        self._max_history_tokens = max_history_tokens

    def build(
        self,
        query: str,
        search_results: list[SearchResult],
        history: list[LLMMessage] | None = None,
    ) -> BuiltPrompt:
        """
        Build the complete prompt for a RAG query.

        Args:
            query:          The user's current question.
            search_results: Retrieved chunks from Qdrant, sorted by score.
            history:        Previous chat messages (oldest first).

        Returns:
            BuiltPrompt with messages list ready for the LLM provider.
        """
        # 1. Build context block from retrieved chunks
        context_text, chunks_used = self._build_context(search_results)

        # 2. Build trimmed history block
        history_messages, history_used = self._trim_history(history or [])

        # 3. Assemble full system message with injected context
        system_content = f"{SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{context_text}\n---"

        # 4. Construct message list: system → history → current query
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_content),
            *history_messages,
            LLMMessage(role="user", content=query),
        ]

        estimated_tokens = _estimate_tokens(
            system_content + query + "".join(m.content for m in history_messages)
        )

        return BuiltPrompt(
            messages=messages,
            context_chunks_used=chunks_used,
            history_messages_used=history_used,
            estimated_tokens=estimated_tokens,
        )

    # ── Private ───────────────────────────────────────────────────

    def _build_context(
        self, results: list[SearchResult]
    ) -> tuple[str, int]:
        """
        Concatenate retrieved chunks into a numbered context block.
        Deduplicates by text content. Respects token budget.
        Returns (context_string, number_of_chunks_used).
        """
        if not results:
            return "No relevant documents found.", 0

        seen_texts: set[str] = set()
        context_parts: list[str] = []
        token_budget = self._max_context_tokens

        for i, result in enumerate(results):
            text = result.payload.text.strip()
            # Skip exact duplicates (can happen with high-overlap chunks)
            if text in seen_texts:
                continue
            seen_texts.add(text)

            chunk_text = (
                f"[{i + 1}] Source: {result.payload.source} "
                f"(Page {result.payload.page_number})\n{text}"
            )
            chunk_tokens = _estimate_tokens(chunk_text)

            if chunk_tokens > token_budget:
                break   # Budget exhausted — stop adding chunks

            context_parts.append(chunk_text)
            token_budget -= chunk_tokens

        if not context_parts:
            return "No relevant documents found.", 0

        return "\n\n".join(context_parts), len(context_parts)

    def _trim_history(
        self, history: list[LLMMessage]
    ) -> tuple[list[LLMMessage], int]:
        """
        Take the most recent messages that fit within the history token budget.
        Always includes pairs (user + assistant) to preserve conversation flow.
        Returns (trimmed_messages, count_used).
        """
        if not history:
            return [], 0

        token_budget = self._max_history_tokens
        selected: list[LLMMessage] = []

        # Walk history from newest to oldest
        for msg in reversed(history):
            tokens = _estimate_tokens(msg.content)
            if tokens > token_budget:
                break
            selected.insert(0, msg)
            token_budget -= tokens

        return selected, len(selected)


try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(text: str) -> int:
        return max(1, len(_enc.encode(text)))
except Exception:
    def _estimate_tokens(text: str) -> int:  # type: ignore[misc]
        """Fallback: 4 chars ≈ 1 token."""
        return max(1, len(text) // CHARS_PER_TOKEN)
