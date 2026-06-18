"""
src/services/answer_validator.py
─────────────────────────────────
Programmatic answer validation — the orchestrator's "evaluator" step.

Deliberately NOT an LLM call (per the latency decision in the agentic
redesign): on a local CPU model an evaluator pass would double response
time and block token streaming. These checks are cheap, deterministic,
and catch the failure modes that actually occur:

1. Hallucinated citations — `[7]` when only 5 sources exist — are
   stripped so the UI never renders a dead chip.
2. Broken abstention contract — retrieval said "abstain" but the model
   answered anyway → replace with the canonical phrase (the system
   prompt already demanded it; this enforces it).
3. Voluntary abstention — model declined even though retrieval was
   confident → surfaced as a warning event, answer left intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.logger import get_logger
from src.services.prompt_builder import NO_ANSWER_PHRASE

log = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class ValidationResult:
    answer: str                       # possibly corrected
    changed: bool = False
    stripped_citations: list[int] = field(default_factory=list)
    warning_code: str | None = None   # "abstention_mismatch" | "model_abstained" | None


def validate_answer(
    answer: str,
    *,
    num_sources: int,
    abstained: bool,
    allow_citations: bool = True,
) -> ValidationResult:
    """
    Validate and (minimally) correct a generated answer.

    Args:
        answer:          The full generated text.
        num_sources:     Length of the context-ordered sources list; a
                         citation [n] is valid iff 1 <= n <= num_sources.
        abstained:       Retrieval-side abstention signal.
        allow_citations: False for summary / off-topic routes where [n]
                         markers are never legitimate.
    """
    stripped: list[int] = []

    def _strip(match: re.Match) -> str:
        n = int(match.group(1))
        if allow_citations and 1 <= n <= num_sources:
            return match.group(0)
        stripped.append(n)
        return ""

    cleaned = _CITATION_RE.sub(_strip, answer)
    if stripped:
        # Tidy whitespace artifacts left by removed markers.
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
        log.warning(
            "validator_stripped_citations",
            stripped=stripped,
            num_sources=num_sources,
        )

    changed = cleaned != answer
    warning_code: str | None = None

    if abstained and NO_ANSWER_PHRASE not in cleaned:
        # Retrieval said "no grounding" but the model answered anyway.
        # Persisted truth must honor the abstention contract.
        log.warning(
            "validator_abstention_enforced",
            original_preview=cleaned[:160],
        )
        cleaned = NO_ANSWER_PHRASE
        changed = True
        warning_code = "abstention_mismatch"
    elif not abstained and cleaned.strip().startswith(NO_ANSWER_PHRASE):
        # Model declined despite confident retrieval — keep its answer
        # but tell the user the corpus likely does contain material.
        warning_code = "model_abstained"

    return ValidationResult(
        answer=cleaned,
        changed=changed,
        stripped_citations=stripped,
        warning_code=warning_code,
    )
