from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    lm_studio_url: str
    embedding_model: str
    embedding_dim: int
    db_path: Path
    chunk_tokens: int
    chunk_overlap_tokens: int
    embed_batch_size: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            lm_studio_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            db_path=Path(os.getenv("DB_PATH", "./mcpdf.db")).expanduser().resolve(),
            chunk_tokens=int(os.getenv("CHUNK_TOKENS", "512")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "64")),
            embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        )
