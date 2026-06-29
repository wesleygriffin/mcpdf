from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _load_env_once() -> None:
    """Load credentials from the user's env file.

    Search order: MCPDF_ENV_FILE if set, then a local .env.
    Each call is no-op-safe; python-dotenv won't clobber values already in os.environ.
    """
    candidates: list[Path] = []
    if env_path := os.environ.get("MCPDF_ENV_FILE"):
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path(".env"))
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
            return


_load_env_once()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing for the requested operation."""


@dataclass(frozen=True)
class Config:
    # Cloudflare (required for upload, optional for extract)
    cf_account_id: str | None
    cf_api_token: str | None
    vectorize_index: str
    d1_database: str
    workers_ai_model: str

    # Hugging Face (required to download the gated Gemma tokenizer)
    hf_token: str | None
    tokenizer_repo: str

    # Chunking
    chunk_tokens: int
    chunk_overlap_tokens: int
    embed_batch_size: int

    @classmethod
    def from_env(cls, *, corpus: str | None = None) -> "Config":
        """Build a Config from environment variables.

        If `corpus` is provided, it overrides both VECTORIZE_INDEX and
        D1_DATABASE — the convention is that a corpus name names both
        resources together (one (vectorize, d1) pair per corpus).
        """
        return cls(
            cf_account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            cf_api_token=os.getenv("CLOUDFLARE_API_TOKEN"),
            vectorize_index=corpus or os.getenv("VECTORIZE_INDEX", "mcpdf"),
            d1_database=corpus or os.getenv("D1_DATABASE", "mcpdf"),
            workers_ai_model=os.getenv("WORKERS_AI_MODEL", "@cf/google/embeddinggemma-300m"),
            hf_token=os.getenv("HF_TOKEN"),
            tokenizer_repo=os.getenv("TOKENIZER_REPO", "google/embeddinggemma-300m"),
            chunk_tokens=int(os.getenv("CHUNK_TOKENS", "1500")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "150")),
            embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        )

    def require_cloudflare(self) -> None:
        missing = [
            name
            for name, val in (
                ("CLOUDFLARE_ACCOUNT_ID", self.cf_account_id),
                ("CLOUDFLARE_API_TOKEN", self.cf_api_token),
            )
            if not val
        ]
        if missing:
            raise ConfigError(
                f"Missing required env vars for Cloudflare upload: {', '.join(missing)}"
            )
