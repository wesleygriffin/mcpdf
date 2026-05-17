from __future__ import annotations

import json
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

_API_BASE = "https://api.cloudflare.com/client/v4"


class CloudflareAPIError(RuntimeError):
    """Cloudflare API returned an error response."""


class CloudflareClient:
    """Thin async REST wrapper over the Cloudflare API surfaces we need.

    Only the calls the indexer's upload step performs: Workers AI embeddings,
    Vectorize bulk insert, and D1 query. Errors raise CloudflareAPIError so the
    CLI can render them with context.
    """

    def __init__(self, account_id: str, api_token: str, *, timeout_s: float = 120.0) -> None:
        self._account_id = account_id
        self._client = httpx.AsyncClient(
            base_url=f"{_API_BASE}/accounts/{account_id}",
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=timeout_s,
        )

    async def __aenter__(self) -> "CloudflareClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        """Call Workers AI embeddings; returns one vector per input text in order."""
        resp = await self._client.post(f"/ai/run/{model}", json={"text": texts})
        payload = _unwrap(resp)
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise CloudflareAPIError(
                f"Workers AI embedding response shape unexpected: "
                f"got {len(data) if isinstance(data, list) else type(data).__name__} "
                f"vectors for {len(texts)} inputs"
            )
        return data

    async def vectorize_insert(self, index_name: str, vectors: list[dict[str, Any]]) -> dict:
        """Bulk-insert vectors into a Vectorize index. NDJSON body."""
        body = "\n".join(json.dumps(v, separators=(",", ":")) for v in vectors).encode()
        resp = await self._client.post(
            f"/vectorize/v2/indexes/{index_name}/insert",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        return _unwrap(resp)

    async def vectorize_delete_by_ids(self, index_name: str, ids: list[str]) -> dict:
        """Delete vectors by ID from a Vectorize index."""
        if not ids:
            return {}
        resp = await self._client.post(
            f"/vectorize/v2/indexes/{index_name}/delete_by_ids",
            json={"ids": ids},
        )
        return _unwrap(resp)

    async def d1_query(
        self, database_id: str, sql: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a single SQL statement against D1; returns the rows of the first result set."""
        resp = await self._client.post(
            f"/d1/database/{database_id}/query",
            json={"sql": sql, "params": params or []},
        )
        payload = _unwrap(resp)
        # D1's API returns a list of result sets (one per statement).
        if isinstance(payload, list) and payload:
            return payload[0].get("results", [])
        return []

    async def d1_database_id_for_name(self, name: str) -> str:
        """Look up a D1 database's UUID by its human-readable name."""
        resp = await self._client.get("/d1/database", params={"name": name})
        items = _unwrap(resp)
        if not isinstance(items, list):
            raise CloudflareAPIError(f"unexpected D1 list response: {items!r}")
        for item in items:
            if item.get("name") == name:
                return item["uuid"]
        raise CloudflareAPIError(f"D1 database {name!r} not found in account")


def _unwrap(resp: httpx.Response) -> Any:
    """Pull `result` out of Cloudflare's standard envelope, or raise with context."""
    if resp.status_code >= 400:
        raise CloudflareAPIError(
            f"{resp.request.method} {resp.request.url.path} -> {resp.status_code}: {resp.text}"
        )
    payload = resp.json()
    if not payload.get("success", False):
        raise CloudflareAPIError(
            f"{resp.request.method} {resp.request.url.path}: {payload.get('errors')}"
        )
    return payload.get("result")
