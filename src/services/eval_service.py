"""
src/services/eval_service.py
──────────────────────────────
RAG evaluation pipeline with RAGAS-inspired metrics.

Metrics implemented:
    faithfulness:       Does the answer contain only claims supported
                        by the retrieved context? (hallucination detection)
                        Score: 0.0 (hallucinated) → 1.0 (fully grounded)

    answer_relevance:   Is the answer actually responsive to the question?
                        Score: 0.0 (off-topic) → 1.0 (fully relevant)

    context_precision:  Of the retrieved chunks, how many were actually
                        useful for generating the answer?
                        Score: 0.0 (all noise) → 1.0 (all useful)

    context_recall:     Did the retrieved chunks contain all information
                        needed to answer correctly?
                        Score: 0.0 (missing context) → 1.0 (complete context)

    answer_similarity:  Semantic similarity between generated answer and
                        a reference answer (when a reference is provided).
                        Score: cosine similarity of embeddings [0, 1]

All metrics are computed using LLM calls (faithfulness, relevance) or
embedding similarity (answer_similarity, context_precision).

Usage:
    service = EvalService(llm=llm, embedder=embedder)

    result = await service.evaluate(
        question="What is the refund policy?",
        answer="Refunds are allowed within 30 days.",
        contexts=["Our policy allows 30-day refunds..."],
        reference="The refund window is 30 days.",
    )
    print(result.faithfulness)     # 0.95
    print(result.answer_relevance) # 0.98
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from src.core.logger import get_logger
from src.providers.embedding.base import BaseEmbeddingProvider
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage

log = get_logger(__name__)

_EVAL_LLM_CONFIG = LLMConfig(temperature=0.0, max_tokens=512)


# ── Prompts ───────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI answer is faithful to the given context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

List each claim made in the ANSWER. For each claim, judge whether it is \
SUPPORTED by the CONTEXT, CONTRADICTED by the CONTEXT, or NOT IN CONTEXT.

Respond in JSON only:
{{"claims": [{{"claim": "...", "verdict": "supported|contradicted|not_in_context"}}]}}"""

_ANSWER_RELEVANCE_PROMPT = """\
You are evaluating whether an AI answer is relevant to the question.

QUESTION: {question}
ANSWER: {answer}

Generate {n_questions} questions that the ANSWER would be a good response to.
Respond in JSON only:
{{"generated_questions": ["...", "..."]}}"""

_CONTEXT_PRECISION_PROMPT = """\
You are evaluating whether retrieved context was useful for answering a question.

QUESTION: {question}
CONTEXT CHUNK: {context_chunk}
ANSWER: {answer}

Was this context chunk useful for generating the answer?
Respond in JSON only: {{"useful": true}} or {{"useful": false}}"""

_CONTEXT_RECALL_PROMPT = """\
You are evaluating whether retrieved context covers all information needed.

QUESTION: {question}
REFERENCE ANSWER: {reference}
RETRIEVED CONTEXT:
{context}

List each statement in the REFERENCE ANSWER. For each, judge whether it \
can be attributed to the RETRIEVED CONTEXT.

Respond in JSON only:
{{"statements": [{{"statement": "...", "attributed": true}}]}}"""


@dataclass
class EvalResult:
    """Results from a full RAG evaluation run."""
    question: str
    answer: str
    faithfulness: float | None = None          # 0-1: no hallucination
    answer_relevance: float | None = None       # 0-1: answers the question
    context_precision: float | None = None      # 0-1: retrieved useful chunks
    context_recall: float | None = None         # 0-1: all needed info retrieved
    answer_similarity: float | None = None      # 0-1: vs reference answer

    # Composite score (average of available metrics)
    @property
    def composite_score(self) -> float:
        scores = [
            s for s in [
                self.faithfulness,
                self.answer_relevance,
                self.context_precision,
                self.context_recall,
            ]
            if s is not None
        ]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer[:200],
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_similarity": self.answer_similarity,
            "composite_score": self.composite_score,
        }


@dataclass
class EvalDatapoint:
    """A single evaluation test case."""
    question: str
    reference_answer: str | None = None
    document_ids: list[str] = field(default_factory=list)


@dataclass
class EvalSuiteResult:
    """Results from running a full evaluation suite."""
    results: list[EvalResult]

    @property
    def mean_faithfulness(self) -> float:
        return _mean([r.faithfulness for r in self.results if r.faithfulness is not None])

    @property
    def mean_answer_relevance(self) -> float:
        return _mean([r.answer_relevance for r in self.results if r.answer_relevance is not None])

    @property
    def mean_context_precision(self) -> float:
        return _mean([r.context_precision for r in self.results if r.context_precision is not None])

    @property
    def mean_composite(self) -> float:
        return _mean([r.composite_score for r in self.results])

    def summary(self) -> dict:
        return {
            "n_questions": len(self.results),
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevance": self.mean_answer_relevance,
            "mean_context_precision": self.mean_context_precision,
            "mean_composite_score": self.mean_composite,
        }


class EvalService:
    """
    Evaluates RAG pipeline quality using LLM-as-judge metrics.

    All metrics are independent and can be computed in parallel.
    Each metric call is wrapped in try/except so one failure doesn't
    block the others — the metric simply stays None.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        embedder: BaseEmbeddingProvider | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str | None = None,
    ) -> EvalResult:
        """
        Run all applicable metrics for a single Q&A pair.

        Args:
            question:   The user question.
            answer:     The generated answer to evaluate.
            contexts:   Retrieved context chunks (plain text list).
            reference:  Optional: gold-standard reference answer.

        Returns:
            EvalResult with individual metric scores and composite.
        """
        result = EvalResult(question=question, answer=answer)
        context_text = "\n\n".join(contexts)

        # All metrics run concurrently
        tasks = {
            "faithfulness": self._faithfulness(question, answer, context_text),
            "answer_relevance": self._answer_relevance(question, answer),
            "context_precision": self._context_precision(question, answer, contexts),
        }
        if reference:
            tasks["context_recall"] = self._context_recall(
                question, reference, context_text
            )
        if reference and self._embedder:
            tasks["answer_similarity"] = self._answer_similarity(answer, reference)

        metric_results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        for key, val in zip(tasks.keys(), metric_results):
            if isinstance(val, Exception):
                log.warning("eval_metric_failed", metric=key, error=str(val))
            else:
                setattr(result, key, val)

        log.info(
            "eval_complete",
            question_preview=question[:50],
            composite=result.composite_score,
            faithfulness=result.faithfulness,
            relevance=result.answer_relevance,
        )
        return result

    # ── Individual Metrics ────────────────────────────────────────

    async def _faithfulness(
        self, question: str, answer: str, context: str
    ) -> float:
        """
        Faithfulness: fraction of answer claims supported by context.
        Uses LLM to decompose answer into claims and verify each.
        """
        prompt = _FAITHFULNESS_PROMPT.format(
            context=context[:2000],
            question=question,
            answer=answer,
        )
        response = await self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            _EVAL_LLM_CONFIG,
        )
        data = _parse_json(response.content)
        if not data or "claims" not in data:
            return 0.5  # Neutral on parse failure

        claims = data["claims"]
        if not claims:
            return 1.0

        supported = sum(
            1 for c in claims
            if c.get("verdict") == "supported"
        )
        return round(supported / len(claims), 4)

    async def _answer_relevance(self, question: str, answer: str) -> float:
        """
        Answer relevance: generate reverse questions and measure similarity.
        High-quality answers generate questions similar to the original.
        """
        n = 3
        prompt = _ANSWER_RELEVANCE_PROMPT.format(
            question=question, answer=answer, n_questions=n
        )
        response = await self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            _EVAL_LLM_CONFIG,
        )
        data = _parse_json(response.content)
        if not data or "generated_questions" not in data:
            return 0.5

        generated = data["generated_questions"]
        if not generated or not self._embedder:
            # Without embedder, use LLM similarity estimate
            return await self._llm_similarity_score(question, generated)

        # Embed original question and generated questions, compute cosine sim
        q_vec = await self._embedder.embed_query(question)
        gen_vecs = await self._embedder.embed_documents(generated)

        sims = [_cosine_sim(q_vec, gv) for gv in gen_vecs]
        return round(sum(sims) / len(sims), 4) if sims else 0.5

    async def _context_precision(
        self, question: str, answer: str, contexts: list[str]
    ) -> float:
        """
        Context precision: fraction of retrieved chunks that were useful.
        """
        if not contexts:
            return 0.0

        verdicts = await asyncio.gather(
            *[
                self._is_chunk_useful(question, answer, chunk)
                for chunk in contexts
            ],
            return_exceptions=True,
        )

        useful = sum(
            1 for v in verdicts
            if not isinstance(v, Exception) and v is True
        )
        return round(useful / len(contexts), 4)

    async def _context_recall(
        self, question: str, reference: str, context: str
    ) -> float:
        """
        Context recall: fraction of reference answer statements that
        are attributable to the retrieved context.
        """
        prompt = _CONTEXT_RECALL_PROMPT.format(
            question=question,
            reference=reference,
            context=context[:2000],
        )
        response = await self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            _EVAL_LLM_CONFIG,
        )
        data = _parse_json(response.content)
        if not data or "statements" not in data:
            return 0.5

        statements = data["statements"]
        if not statements:
            return 1.0

        attributed = sum(1 for s in statements if s.get("attributed") is True)
        return round(attributed / len(statements), 4)

    async def _answer_similarity(
        self, answer: str, reference: str
    ) -> float:
        """
        Answer similarity: cosine similarity between answer and reference
        in embedding space.
        """
        if not self._embedder:
            return 0.5

        vecs = await self._embedder.embed_documents([answer, reference])
        if len(vecs) < 2:
            return 0.5
        return round(_cosine_sim(vecs[0], vecs[1]), 4)

    async def _is_chunk_useful(
        self, question: str, answer: str, chunk: str
    ) -> bool:
        prompt = _CONTEXT_PRECISION_PROMPT.format(
            question=question,
            context_chunk=chunk[:600],
            answer=answer,
        )
        response = await self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            LLMConfig(temperature=0.0, max_tokens=50),
        )
        data = _parse_json(response.content)
        return bool(data and data.get("useful") is True)

    async def _llm_similarity_score(
        self, question: str, generated: list[str]
    ) -> float:
        """Fallback: ask LLM to rate question similarity as 0-100."""
        prompt = (
            f"Rate how semantically similar these questions are to the original "
            f"on a scale of 0-100. Respond with ONLY an integer.\n\n"
            f"Original: {question}\n"
            f"Generated: {'; '.join(generated[:3])}"
        )
        try:
            resp = await self._llm.generate(
                [LLMMessage(role="user", content=prompt)],
                LLMConfig(temperature=0.0, max_tokens=10),
            )
            match = re.search(r"\d+", resp.content)
            if match:
                return min(1.0, int(match.group()) / 100)
        except Exception:
            pass
        return 0.5


# ── Helpers ───────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    """Parse JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity between two pre-normalised vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0