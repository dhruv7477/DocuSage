"""
src/api/v1/eval.py
───────────────────
Evaluation API endpoints.

POST /eval/run           Run evaluation on a single Q&A pair
POST /eval/suite         Run evaluation on a batch of test cases
GET  /eval/results/{id}  Retrieve stored eval results (Phase 5: from DB)

These endpoints are intended for workspace admins and developers to
measure RAG quality and track regression across pipeline changes.

Authentication: all endpoints require a valid JWT token.
"""

import uuid

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from src.dependencies import require_admin
from src.models.user import User
from src.providers.embedding.factory import get_embedding_provider
from src.providers.llm.factory import get_llm_provider
from src.services.eval_service import EvalResult, EvalService, EvalSuiteResult

router = APIRouter(prefix="/eval", tags=["Evaluation"])


# ── Request / Response Schemas ────────────────────────────────────


class EvalRunRequest(BaseModel):
    """Evaluate a single Q&A pair."""

    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=8000)
    contexts: list[str] = Field(min_length=1, max_length=20)
    reference_answer: str | None = None
    workspace_id: uuid.UUID | None = None


class EvalSuiteRequest(BaseModel):
    """Evaluate a batch of Q&A pairs."""

    workspace_id: uuid.UUID
    test_cases: list[EvalRunRequest] = Field(min_length=1, max_length=50)


class EvalResultResponse(BaseModel):
    question: str
    answer: str
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    context_recall: float | None
    answer_similarity: float | None
    composite_score: float


class EvalSuiteResponse(BaseModel):
    n_questions: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_precision: float
    mean_composite_score: float
    results: list[EvalResultResponse]


# ── Helpers ───────────────────────────────────────────────────────


def _result_to_response(r: EvalResult) -> EvalResultResponse:
    return EvalResultResponse(
        question=r.question,
        answer=r.answer,
        faithfulness=r.faithfulness,
        answer_relevance=r.answer_relevance,
        context_precision=r.context_precision,
        context_recall=r.context_recall,
        answer_similarity=r.answer_similarity,
        composite_score=r.composite_score,
    )


def _get_eval_service() -> EvalService:
    return EvalService(
        llm=get_llm_provider(),
        embedder=get_embedding_provider(),
    )


# ── Endpoints ─────────────────────────────────────────────────────


@router.post(
    "/run",
    response_model=EvalResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a single Q&A pair with RAGAS-style metrics (admin only)",
)
async def run_eval(
    payload: EvalRunRequest,
    current_user: User = Depends(require_admin),
    eval_service: EvalService = Depends(_get_eval_service),
) -> EvalResultResponse:
    """
    Compute faithfulness, answer relevance, and context precision
    for a single generated answer.

    Optionally compute context recall and answer similarity when
    a reference answer is provided.
    """
    result = await eval_service.evaluate(
        question=payload.question,
        answer=payload.answer,
        contexts=payload.contexts,
        reference=payload.reference_answer,
    )
    return _result_to_response(result)


@router.post(
    "/suite",
    response_model=EvalSuiteResponse,
    status_code=status.HTTP_200_OK,
    summary="Run evaluation suite on a batch of Q&A pairs (admin only)",
)
async def run_eval_suite(
    payload: EvalSuiteRequest,
    current_user: User = Depends(require_admin),
    eval_service: EvalService = Depends(_get_eval_service),
) -> EvalSuiteResponse:
    """
    Evaluate multiple Q&A pairs and return aggregate statistics.
    Returns individual results plus mean scores per metric.
    """
    import asyncio

    results = await asyncio.gather(
        *[
            eval_service.evaluate(
                question=tc.question,
                answer=tc.answer,
                contexts=tc.contexts,
                reference=tc.reference_answer,
            )
            for tc in payload.test_cases
        ],
        return_exceptions=True,
    )

    eval_results = [r for r in results if not isinstance(r, Exception)]

    suite = EvalSuiteResult(results=eval_results)
    return EvalSuiteResponse(
        **suite.summary(),
        results=[_result_to_response(r) for r in eval_results],
    )
