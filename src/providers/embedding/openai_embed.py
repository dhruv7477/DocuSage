"""
src/providers/embedding/openai_embed.py
─────────────────────────────────────────
OpenAI embedding provider — plug-in for text-embedding-3-large.

Activation: set EMBEDDING_PROVIDER=openai and OPENAI_API_KEY in .env.
Dimension: 3072 (text-embedding-3-large) or 1536 (text-embedding-3-small).

Note: If switching from BGE (1024-dim) to OpenAI (3072-dim), you MUST
recreate the Qdrant collection — dimensions are fixed at collection creation.
"""

from openai import AsyncOpenAI, APIConnectionError, APIStatusError

from src.core.exceptions import EmbeddingError
from src.core.logger import get_logger
from src.providers.embedding.base import BaseEmbeddingProvider

log = get_logger(__name__)

_MODEL_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embedding provider.")
        self._model = model
        self._dim = _MODEL_DIMENSIONS.get(model, 3072)
        self._client = AsyncOpenAI(api_key=api_key, max_retries=2)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return f"openai-embed:{self._model}"

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI supports up to 2048 inputs per request; we batch at 512
        batch_size = 512
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = await self._client.embeddings.create(
                    model=self._model, input=batch
                )
            except APIConnectionError as exc:
                raise EmbeddingError("Cannot reach OpenAI Embeddings.", detail=str(exc)) from exc
            except APIStatusError as exc:
                raise EmbeddingError(f"OpenAI Embeddings error {exc.status_code}", detail=exc.message) from exc

            # Sort by index to preserve order (OpenAI doesn't guarantee order)
            items = sorted(resp.data, key=lambda x: x.index)
            all_vectors.extend([item.embedding for item in items])
        return all_vectors

    async def health(self) -> bool:
        try:
            await self._client.embeddings.create(model=self._model, input=["ok"])
            return True
        except Exception:
            return False
