"""
src/api/v1/demo.py
──────────────────
Public, unauthenticated facts about this deployment's demo mode, so the
sign-up page can tell visitors what they agree to before they create an
account.
"""

from urllib.parse import urlsplit

from fastapi import APIRouter
from pydantic import BaseModel

from src.configs.config import Settings, get_settings

router = APIRouter(prefix="/demo", tags=["Demo"])


class DemoInfo(BaseModel):
    enabled: bool
    max_documents: int  # 0 = unlimited
    max_upload_mb: int  # 0 = unlimited
    questions_per_day: int  # per user; 0 = unlimited
    site_questions_per_day: int  # all users together; 0 = unlimited
    wipe_hour_utc: int | None  # uploads are deleted daily at this hour
    llm_host: str | None  # third party that receives questions and excerpts


def _llm_host(s: Settings) -> str | None:
    if s.llm_provider == "openai":
        return urlsplit(s.openai_base_url).hostname or "api.openai.com"
    if s.llm_provider == "vllm":
        return urlsplit(s.vllm_base_url).hostname
    return None  # local Ollama: nothing leaves the server


@router.get("", response_model=DemoInfo, summary="Demo limits and data handling")
async def demo_info() -> DemoInfo:
    s = get_settings()
    return DemoInfo(
        enabled=s.demo_mode,
        max_documents=s.quota_max_documents,
        max_upload_mb=s.quota_max_upload_mb,
        questions_per_day=s.quota_questions_per_day,
        site_questions_per_day=s.demo_global_questions_per_day,
        wipe_hour_utc=s.demo_wipe_hour_utc if s.demo_mode else None,
        llm_host=_llm_host(s),
    )
