from __future__ import annotations

from typing import Literal, Sequence

import httpx

PrefixMode = Literal["document", "query", "none"]

# Nomic-family models expect these task prefixes. Stripped automatically if the
# model doesn't need them via the NOMIC_PREFIX_DISABLE env (handled in caller).
_PREFIXES: dict[PrefixMode, str] = {
    "document": "search_document: ",
    "query": "search_query: ",
    "none": "",
}


class LMStudioEmbeddingClient:
    """Thin async client over LM Studio's OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str, model: str, *, timeout_s: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LMStudioEmbeddingClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def embed(
        self,
        texts: Sequence[str],
        *,
        prefix: PrefixMode,
    ) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [_PREFIXES[prefix] + t for t in texts]
        resp = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": prefixed},
        )
        resp.raise_for_status()
        payload = resp.json()
        # OpenAI shape: { data: [{ embedding: [...], index: N }, ...] }
        # Order isn't guaranteed by spec, so sort by index.
        items = sorted(payload["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in items]
