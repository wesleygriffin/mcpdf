from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from tokenizers import Tokenizer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GemmaTokenizer:
    """Thin wrapper over the EmbeddingGemma tokenizer.

    Loads the gated Gemma `tokenizer.json` from Hugging Face once, caches the
    file under the standard HF cache dir, and exposes the encode/decode/count
    operations chunking needs. SentencePiece-based; `add_special_tokens=False`
    so we don't insert BOS/EOS into mid-document chunks.
    """

    tokenizer: Tokenizer

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, repo_id: str, hf_token: str | None) -> "GemmaTokenizer":
        try:
            path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
        except GatedRepoError as exc:
            raise RuntimeError(
                f"{repo_id} is a gated Hugging Face repo. Accept the license at "
                f"https://huggingface.co/{repo_id} and set HF_TOKEN."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Tokenizer repo {repo_id} not found. Check TOKENIZER_REPO."
            ) from exc
        log.debug("Loaded tokenizer from %s", path)
        return cls(tokenizer=Tokenizer.from_file(path))

    def encode_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode_ids(text))
