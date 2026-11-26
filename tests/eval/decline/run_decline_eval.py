"""
tests/eval/decline/run_decline_eval.py
--------------------------------------
Measures the chat pipeline's decline decision with zero LLM calls.

For every question in `questions.jsonl` it calls the app's real
orchestrator (`_get_orchestrator(_get_rag_service())` from
`src.api.v1.chat`) up to the point where tokens would start flowing, and
reads the plan's intent: "off_topic" means the fixed decline reply,
anything else means the question would be answered. The token iterator
is never consumed, so no answer is generated.

Every LLM provider the factory builds is replaced by a guard that counts
and refuses calls. A healthy run reports 0 LLM calls; any other count is
a failure (exit code 1), because it means a code path would spend tokens
before the decline decision.

Questions are single turn (no history), so the standalone-question
rewrite never runs, and the router's LLM tier is off unless
ROUTER_USE_LLM=true.

For questions with an `answer` it also measures recall (where the first
chunk whose page text states the answer ranks in dense search and in the
prompt's context) and citation accuracy under the app's citation rule.
A general-knowledge question must reach the model, and an answer that
cites nothing must be shown with the "Not from your documents" label and
no sources.

It also reports the estimated prompt tokens of every question that reaches
the answer LLM (the cost side), and, in a small workspace (one whose
chunks all fit the prompt), checks that each document's first chunk is in
the prompt.

Run inside the api image (see README.md in this folder):

    python /work/run_decline_eval.py --email decline-eval@example.com
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from sqlalchemy import select

from src.api.v1 import chat as chat_api
from src.configs.config import get_settings
from src.core.logger import configure_logging
from src.db.session import AsyncSessionLocal
from src.models.document import Document, DocumentStatus
from src.models.user import User
from src.models.workspace import Workspace
from src.providers.llm import factory as llm_factory
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage, LLMResponse
from src.services.citation_grounding import NOT_FROM_DOCUMENTS, finalize_rag_answer
from src.services.prompt_builder import NO_ANSWER_PHRASE
from src.services.rag_service import CitedSource
from src.vector_store.factory import get_vector_store

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.jsonl"
WORKSPACE_NAMES = {
    "resume": "Decline eval: resume",
    "resume2": "Decline eval: resume 2",
    "samples": "Sample documents",
}
DECLINED_INTENT = "off_topic"
SUMMARY_INTENT = "doc_summary"
CATEGORY_ORDER = (
    "answerable_specific",
    "answerable_vague",
    "summary",
    "near_topic",
    "off_topic",
    "general_knowledge",
)
MUST_DECLINE = frozenset({"near_topic", "off_topic"})
GENERAL_KNOWLEDGE = "general_knowledge"
# An answer from outside the documents, as a model would write it: no [n].
UNCITED_ANSWER = "In general, this depends on the employer and the role."
SETTINGS_SHOWN = (
    "reranker",
    "reranker_model",
    "abstention_reranker_threshold",
    "abstention_cosine_threshold",
    "retrieval_candidate_k",
    "rag_top_k",
    "embedding_model",
    "enable_hyde",
    "enable_question_router",
    "router_use_llm",
    "doc_summary_enabled",
)


# LLM guard


class LLMCallGuard:
    """Counts attempted LLM calls and makes each one fail loudly."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def trip(self, model: str, method: str) -> NoReturn:
        """Record the attempt, then raise so no request is ever sent."""
        self.calls.append(f"{model}.{method}")
        raise RuntimeError(f"LLM call blocked by the decline eval: {model}.{method}")

    def install(self) -> None:
        """Make the LLM factory build guarded providers from now on."""
        llm_factory._build_llm_provider = lambda model: GuardedLLM(self, model)
        llm_factory.reset_llm_provider()


class GuardedLLM(BaseLLMProvider):
    """An LLM provider whose every call is refused by an `LLMCallGuard`."""

    def __init__(self, guard: LLMCallGuard, model: str) -> None:
        self._guard = guard
        self._model = model

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
        *,
        timeout: float | None = None,
    ) -> LLMResponse:
        self._guard.trip(self._model, "generate")

    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        self._guard.trip(self._model, "stream")
        yield ""

    async def health(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "decline-eval-guard"


# Instrumentation (diagnostics only; the verdict is plan.intent)


@dataclass
class Probe:
    """Captures the reranked outcome and the dense hits of one question."""

    outcome: Any = None
    dense_hits: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self.outcome = None
        self.dense_hits = []


def _instrument(rag: Any, probe: Probe) -> None:
    """Wrap the RAG service and vector store so each question's scores are kept."""
    prepare_stream = rag.prepare_stream

    async def recording_prepare(*args: Any, **kwargs: Any) -> Any:
        outcome, tokens = await prepare_stream(*args, **kwargs)
        probe.outcome = outcome
        return outcome, tokens

    rag.prepare_stream = recording_prepare

    store = get_vector_store()
    search = store.search

    async def recording_search(*args: Any, **kwargs: Any) -> Any:
        hits = await search(*args, **kwargs)
        probe.dense_hits.extend(hits)
        return hits

    store.search = recording_search


# Data loading


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    workspace: str
    question: str
    doc: str | None = None
    note: str = ""
    split: str = "tune"  # tune | holdout (stratified by category, drawn once)
    # Entity lookups: the exact text a cited chunk must contain.
    answer: str | None = None

    @property
    def expected(self) -> str:
        return "decline" if self.category in MUST_DECLINE else "answer"


@dataclass(frozen=True)
class WorkspaceContext:
    id: uuid.UUID
    documents: list[tuple[str, str]]  # (document id, original filename)


def load_questions(path: Path) -> list[Question]:
    """Read one JSON object per non-empty line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


async def load_context(email: str) -> tuple[User, dict[str, WorkspaceContext]]:
    """Fetch the eval user and its workspaces' completed documents."""
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == email.lower()))
        if user is None:
            raise SystemExit(f"No account {email}; run setup_account.py first.")
        spaces: dict[str, WorkspaceContext] = {}
        for key, name in WORKSPACE_NAMES.items():
            workspace = await db.scalar(
                select(Workspace).where(
                    Workspace.owner_id == user.id, Workspace.name == name
                )
            )
            if workspace is None:
                raise SystemExit(f"Workspace {name!r} missing for {email}.")
            docs = await db.scalars(
                select(Document)
                .where(
                    Document.workspace_id == workspace.id,
                    Document.user_id == user.id,
                    Document.status == DocumentStatus.COMPLETED,
                )
                .order_by(Document.created_at.desc())
            )
            documents = [(str(d.id), d.original_filename) for d in docs]
            if not documents:
                raise SystemExit(f"Workspace {name!r} has no completed documents.")
            spaces[key] = WorkspaceContext(workspace.id, documents)
    return user, spaces


# Evaluation


@dataclass
class Result:
    id: str
    category: str
    split: str
    workspace: str
    question: str
    expected: str
    outcome: str  # answer | decline | error
    intent: str
    routes: list[str]
    warnings: list[str]
    confidence: float | None
    top3: list[float]
    top_dense: float | None
    seconds: float
    error: str = ""
    # Recall and citation accuracy, for questions with an `answer`: the
    # 1-based rank of the first chunk whose page text states it, in dense
    # search and in the prompt's context (None: not there).
    answer_dense_rank: int | None = None
    answer_context_rank: int | None = None
    citations_checked: int = 0
    citation_misses: list[int] = field(default_factory=list)
    # General knowledge: an uncited answer is labelled and shows no sources.
    uncited_labelled: bool | None = None
    # Cost and context: the estimated prompt tokens of a question that
    # reaches the answer LLM, whether its workspace counts as small, and
    # whether every document's first chunk is in the prompt's context.
    prompt_tokens: int | None = None
    small_workspace: bool | None = None
    first_chunks_in_context: bool | None = None

    @property
    def answer_in_context(self) -> bool | None:
        if self.citations_checked == 0 and self.answer_context_rank is None:
            return None
        return self.answer_context_rank is not None

    @property
    def correct(self) -> bool:
        return self.outcome == self.expected


async def evaluate(
    question: Question,
    orchestrator: Any,
    user: User,
    space: WorkspaceContext,
    probe: Probe,
) -> Result:
    """Run one question through `prepare()` and classify the plan."""
    events: list[dict] = []
    emit: Callable[[dict], None] = events.append
    probe.reset()
    started = time.perf_counter()
    intent, error = "", ""
    citations: tuple[int, list[int]] = (0, [])
    context_rank: int | None = None
    uncited_labelled: bool | None = None
    first_chunks: bool | None = None
    try:
        plan = await orchestrator.prepare(
            query=question.question,
            workspace_id=space.id,
            user=user,
            history_messages=None,
            document_ids=None,
            workspace_documents=space.documents,
            emit=emit,
        )
        intent = plan.intent
        if question.answer and intent == "rag_query":
            context_rank = _first_rank([s.text for s in plan.sources], question.answer)
            citations = check_citations(question, plan.sources)
        if question.category == GENERAL_KNOWLEDGE and intent == "rag_query":
            uncited_labelled = check_uncited(question, plan.sources)
        if intent == "rag_query":
            first_chunks = _first_chunks_in(plan.sources, space)
        await _close(plan.token_iter)
    except Exception as exc:  # reported per question, never hidden
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started

    outcome = probe.outcome
    reranked = list(getattr(outcome, "results", None) or [])
    if error:
        verdict = "error"
    else:
        verdict = "decline" if intent == DECLINED_INTENT else "answer"
    prompt = getattr(outcome, "prompt", None)
    return Result(
        id=question.id,
        category=question.category,
        split=question.split,
        workspace=question.workspace,
        question=question.question,
        expected=question.expected,
        outcome=verdict,
        intent=intent,
        routes=[e["intent"] for e in events if e.get("type") == "route"],
        warnings=[e.get("code", "") for e in events if e.get("type") == "warning"],
        confidence=getattr(outcome, "confidence", None),
        top3=[round(r.score, 4) for r in reranked[:3]],
        top_dense=max((h.score for h in probe.dense_hits), default=None),
        seconds=round(elapsed, 2),
        error=error,
        answer_dense_rank=_first_rank(
            [h.payload.text for h in probe.dense_hits], question.answer
        ),
        answer_context_rank=context_rank,
        citations_checked=citations[0],
        citation_misses=citations[1],
        uncited_labelled=uncited_labelled,
        prompt_tokens=(
            prompt.estimated_tokens
            if prompt is not None and intent == "rag_query"
            else None
        ),
        small_workspace=getattr(outcome, "small_workspace", None),
        first_chunks_in_context=first_chunks,
    )


def check_citations(
    question: Question, sources: list[CitedSource]
) -> tuple[int, list[int]]:
    """
    Citation accuracy with no LLM call. A model that cites ANY included
    chunk for the answer (simulated as "<answer> [k].", k = 1..n) must end
    up, after the app's citation rule (`finalize_rag_answer`), showing only
    citations to chunks whose page text contains the answer. When no
    included chunk has it, the answer must become the decline instead.
    Returns (citations checked, k values that missed).
    """
    answer = question.answer or ""
    holds = [_contains(s.text, answer) for s in sources]
    misses: list[int] = []
    for k in range(1, len(sources) + 1):
        final = finalize_rag_answer(
            f"{answer} [{k}].", sources, question=question.question
        )
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", final.answer)]
        if any(holds):
            ok = bool(cited) and all(holds[n - 1] for n in cited)
        else:
            ok = final.answer == NO_ANSWER_PHRASE and not final.sources
        if not ok:
            misses.append(k)
    return len(sources), misses


def check_uncited(question: Question, sources: list[CitedSource]) -> bool:
    """An answer citing nothing is labelled and shown with no sources."""
    final = finalize_rag_answer(UNCITED_ANSWER, sources, question=question.question)
    return final.label == NOT_FROM_DOCUMENTS and not final.sources


def _first_chunks_in(sources: list[CitedSource], space: WorkspaceContext) -> bool:
    """True when the first chunk of every document in the workspace is a source."""
    present = {s.document_id for s in sources if s.chunk_index == 0}
    return all(doc_id in present for doc_id, _ in space.documents)


def _first_rank(texts: list[str], answer: str | None) -> int | None:
    """1-based position of the first text that contains `answer`."""
    if not answer:
        return None
    return next((i for i, t in enumerate(texts, 1) if _contains(t, answer)), None)


def _contains(text: str, answer: str) -> bool:
    """Case- and whitespace-insensitive substring test."""
    return " ".join(answer.lower().split()) in " ".join(text.lower().split())


async def _close(token_iter: Any) -> None:
    """Close the unconsumed token iterator so no generator is left pending."""
    aclose = getattr(token_iter, "aclose", None)
    if aclose is not None:
        await aclose()


# Reporting


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_report(results: list[Result], llm_calls: list[str]) -> None:
    """Print the per-question table, then decline rates per category."""
    print(
        "\n| id | cat | ws | question | route | conf | top-3 | dense | outcome | ok |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        route = ">".join(r.routes)
        top3 = ", ".join(f"{s:.3f}" for s in r.top3) or "-"
        mark = "yes" if r.correct else "NO"
        print(
            f"| {r.id} | {r.category} | {r.workspace} | {r.question} | {route} "
            f"| {_fmt(r.confidence)} | {top3} | {_fmt(r.top_dense)} "
            f"| {r.outcome} | {mark} |"
        )
        if r.error:
            print(f"|   | error: {r.error} |||||||||")

    for label, keep in (
        ("all", lambda r: True),
        ("tune", lambda r: r.split == "tune"),
        ("holdout", lambda r: r.split == "holdout"),
    ):
        _print_summary(label, [r for r in results if keep(r)])
    checked = [r for r in results if r.citations_checked]
    if checked:
        total = sum(r.citations_checked for r in checked)
        missed = sum(len(r.citation_misses) for r in checked)
        in_context = sum(bool(r.answer_in_context) for r in checked)
        print(
            f"\nRecall (entity lookups): answer in context {in_context} of "
            f"{len(checked)}; rank of the first chunk stating it (dense / context):"
        )
        for r in checked:
            print(
                f"  {r.id}: dense {r.answer_dense_rank or '-'} / context "
                f"{r.answer_context_rank or '-'}"
            )
        print(
            f"Citation accuracy (every included chunk cited in turn): "
            f"{total - missed} of {total} shown citations land on a chunk that "
            f"states the answer"
        )
        for r in checked:
            if r.citation_misses:
                print(f"  {r.id}: missed citations: {r.citation_misses}")
    general = [r for r in results if r.category == GENERAL_KNOWLEDGE]
    if general:
        labelled = sum(bool(r.uncited_labelled) for r in general)
        print(
            f"General knowledge: {labelled} of {len(general)} uncited answers "
            f"labelled with no sources"
        )
    _print_cost(results)
    errors = sum(r.outcome == "error" for r in results)
    print(f"\nErrors: {errors}   LLM calls attempted: {len(llm_calls)}")
    if llm_calls:
        print("Blocked LLM calls: " + ", ".join(llm_calls))


def _print_cost(results: list[Result]) -> None:
    """Estimated prompt tokens per workspace, and first chunks in the context."""
    priced = [r for r in results if r.prompt_tokens is not None]
    if not priced:
        return
    total = sum(r.prompt_tokens or 0 for r in priced)
    print(
        f"\nPrompt tokens (estimated) of the {len(priced)} questions that reach "
        f"the answer LLM: {total}, mean {total / len(priced):.0f}"
    )
    for workspace in sorted({r.workspace for r in priced}):
        rows = [r for r in priced if r.workspace == workspace]
        tokens = sum(r.prompt_tokens or 0 for r in rows)
        small = {r.small_workspace for r in rows}
        first = sum(bool(r.first_chunks_in_context) for r in rows)
        print(
            f"  {workspace}: {len(rows)} questions, mean {tokens / len(rows):.0f}, "
            f"small workspace {small}, every first chunk in the context "
            f"{first} of {len(rows)}"
        )


def _print_summary(label: str, results: list[Result]) -> None:
    """Decline rate per category, then the error counts that matter.

    A false decline is an answerable or summary question that got the fixed
    reply. A near-topic or off-topic question that is not declined either
    reaches the LLM, whose grounded prompt lets it decline (this harness
    cannot see that verdict), or goes to the summary route, which never
    declines. Every question that is not declined reaches the LLM.
    """
    if not results:
        return
    print(f"\n[{label}] {len(results)} questions")
    print("| category | n | declined | decline rate | correct |")
    print("|---|---|---|---|---|")
    by_category: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    for category in sorted(by_category, key=_category_rank):
        rows = by_category[category]
        declined = sum(r.outcome == "decline" for r in rows)
        correct = sum(r.correct for r in rows)
        print(
            f"| {category} | {len(rows)} | {declined} | "
            f"{declined / len(rows):.0%} | {correct}/{len(rows)} |"
        )
    false_declines = [
        r.id for r in results if r.expected == "answer" and r.outcome == "decline"
    ]
    to_summary = [
        r.id for r in results if r.expected == "decline" and r.intent == SUMMARY_INTENT
    ]
    to_llm = [
        r.id
        for r in results
        if r.expected == "decline"
        and r.outcome == "answer"
        and r.intent != SUMMARY_INTENT
    ]
    reach = sum(r.outcome == "answer" for r in results)
    print(f"False declines: {len(false_declines)} {false_declines}")
    print(f"Must-decline sent to the summary route: {len(to_summary)} {to_summary}")
    print(f"Must-decline sent to the LLM, which decides: {len(to_llm)} {to_llm}")
    print(f"Reach the LLM: {reach} of {len(results)} ({reach / len(results):.0%})")


def regressions(results: list[Result]) -> list[str]:
    """
    What the decline decision alone must get right, checked in CI with no
    LLM call: no answerable or summary question is declined, and no
    near-topic or off-topic request gets a summary of the user's documents.
    """
    found: list[str] = []
    for r in results:
        if r.outcome == "error":
            found.append(f"{r.id}: error: {r.error}")
        elif r.expected == "answer" and r.outcome == "decline":
            found.append(f"{r.id}: an answerable question was declined: {r.question}")
        elif r.expected == "decline" and r.intent == SUMMARY_INTENT:
            found.append(f"{r.id}: sent to the summary route: {r.question}")
        if r.answer_in_context is False:
            found.append(f"{r.id}: no included chunk states the answer")
        elif r.citation_misses:
            found.append(
                f"{r.id}: citations to {r.citation_misses} do not land on a chunk "
                f"that states the answer"
            )
        if r.uncited_labelled is False:
            found.append(f"{r.id}: an uncited answer is not labelled")
        if r.small_workspace and r.first_chunks_in_context is False:
            found.append(f"{r.id}: a document's first chunk is not in the context")
    return found


def _category_rank(category: str) -> int:
    return CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else 99


# Entry point


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decline-decision eval, no LLM.")
    parser.add_argument("--email", default="decline-eval@example.com")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--json", type=Path, help="also write the results here")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    """Evaluate every question; return the process exit code."""
    guard = LLMCallGuard()
    guard.install()

    settings = get_settings()
    configure_logging(settings.log_level)
    print(
        "Settings: "
        + ", ".join(f"{k}={getattr(settings, k, '-')}" for k in SETTINGS_SHOWN)
    )

    questions = load_questions(args.questions)
    user, spaces = await load_context(args.email)
    rag = chat_api._get_rag_service()
    orchestrator = chat_api._get_orchestrator(rag)
    probe = Probe()
    _instrument(rag, probe)

    results = [
        await evaluate(q, orchestrator, user, spaces[q.workspace], probe)
        for q in questions
    ]
    print_report(results, guard.calls)

    if args.json:
        payload = {
            "settings": {k: getattr(settings, k, None) for k in SETTINGS_SHOWN},
            "llm_calls": guard.calls,
            "results": [
                asdict(r)
                | {"correct": r.correct, "answer_in_context": r.answer_in_context}
                for r in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    found = regressions(results)
    print(f"\nRegressions: {len(found)}")
    for line in found:
        print(f"  {line}")
    return 1 if guard.calls or found else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(_parse_args())))
