from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from .cloudflare import CloudflareAPIError, CloudflareClient
from .config import Config, ConfigError
from .indexer import ChunkRecord, ExtractFailure, collect_paths, extract_chunks
from .tokenizer import GemmaTokenizer

log = logging.getLogger("mcpdf")


def _vector_id(document_path: str, chunk_index: int) -> str:
    """Stable per-(path, chunk_index) vector ID. Re-uploads upsert in place."""
    h = hashlib.sha256(document_path.encode("utf-8")).hexdigest()[:16]
    return f"{h}:{chunk_index:05d}"


def _cmd_extract(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    if not cfg.hf_token:
        log.warning(
            "HF_TOKEN not set — tokenizer download will only work if you've run "
            "`huggingface-cli login` previously."
        )
    tokenizer = GemmaTokenizer.load(cfg.tokenizer_repo, cfg.hf_token)
    paths = collect_paths(args.path, recursive=not args.no_recursive)
    if not paths:
        log.error("No PDFs found at %s", args.path)
        return 2

    log.info("Extracting %d PDF(s) into %s", len(paths), args.out)
    seen_docs: set[str] = set()
    chunk_count = 0
    failures: list[ExtractFailure] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for item in extract_chunks(paths, tokenizer, cfg):
            if isinstance(item, ExtractFailure):
                failures.append(item)
                continue
            f.write(json.dumps(dataclasses.asdict(item), ensure_ascii=False) + "\n")
            chunk_count += 1
            seen_docs.add(item.document_path)

    log.info(
        "Done. Docs: %d, chunks: %d, failures: %d",
        len(seen_docs),
        chunk_count,
        len(failures),
    )
    for fail in failures:
        log.error("FAIL %s: %s", fail.path, fail.error)
    return 1 if failures else 0


def _load_chunks(path: Path) -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ChunkRecord(**json.loads(line)))
    return out


def _group_by_doc(records: list[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    groups: dict[str, list[ChunkRecord]] = defaultdict(list)
    for r in records:
        groups[r.document_path].append(r)
    for chunks in groups.values():
        chunks.sort(key=lambda r: r.chunk_index)
    return groups


async def _upload_one_doc(
    client: CloudflareClient,
    cfg: Config,
    chunks: list[ChunkRecord],
    db_id: str,
    *,
    dry_run: bool,
) -> None:
    doc = chunks[0]
    log.info(
        "%s: %d chunks (%s, %d pages)",
        doc.document_title,
        len(chunks),
        doc.document_path,
        doc.document_total_pages,
    )
    if dry_run:
        return

    # Embed in batches and build Vectorize records.
    vectors: list[dict] = []
    for start in range(0, len(chunks), cfg.embed_batch_size):
        batch = chunks[start : start + cfg.embed_batch_size]
        embeddings = await client.embed_batch(cfg.workers_ai_model, [c.text for c in batch])
        for c, vec in zip(batch, embeddings):
            vectors.append(
                {
                    "id": _vector_id(c.document_path, c.chunk_index),
                    "values": vec,
                    "metadata": {
                        "document_path": c.document_path,
                        "document_title": c.document_title,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "text": c.text,
                    },
                }
            )
    await client.vectorize_insert(cfg.vectorize_index, vectors)

    # Upsert the documents row. SQLite-flavored UPSERT — D1 is SQLite.
    await client.d1_query(
        db_id,
        """
        INSERT INTO documents (path, title, content_sha256, total_pages, chunk_count, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            title = excluded.title,
            content_sha256 = excluded.content_sha256,
            total_pages = excluded.total_pages,
            chunk_count = excluded.chunk_count,
            indexed_at = excluded.indexed_at
        """,
        [
            doc.document_path,
            doc.document_title,
            doc.document_sha256,
            doc.document_total_pages,
            len(chunks),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ],
    )


async def _amain_upload(args: argparse.Namespace) -> int:
    cfg = Config.from_env(corpus=args.corpus)
    try:
        cfg.require_cloudflare()
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    log.info("Target corpus: vectorize=%s d1=%s", cfg.vectorize_index, cfg.d1_database)

    records = _load_chunks(args.chunks)
    if not records:
        log.error("No chunks in %s", args.chunks)
        return 2
    groups = _group_by_doc(records)
    total_tokens = sum(r.token_count for r in records)
    log.info(
        "Loaded %d chunks across %d documents (%s tokens total)",
        len(records),
        len(groups),
        f"{total_tokens:,}",
    )

    async with CloudflareClient(cfg.cf_account_id, cfg.cf_api_token) as client:
        if args.dry_run:
            db_id = "DRY-RUN"
        else:
            db_id = await client.d1_database_id_for_name(cfg.d1_database)
            log.info("Resolved D1 %s -> %s", cfg.d1_database, db_id)
        for path, chunks in groups.items():
            try:
                await _upload_one_doc(client, cfg, chunks, db_id, dry_run=args.dry_run)
            except CloudflareAPIError as exc:
                log.error("FAIL %s: %s", path, exc)
                return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcpdf-index")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract & chunk PDFs into a JSONL file")
    p_extract.add_argument("path", type=Path, help="PDF file or directory")
    p_extract.add_argument(
        "--out", type=Path, default=Path("chunks.jsonl"), help="Output JSONL path"
    )
    p_extract.add_argument("--no-recursive", action="store_true")

    p_upload = sub.add_parser(
        "upload", help="Embed chunks via Workers AI and insert into Vectorize + D1"
    )
    p_upload.add_argument("chunks", type=Path, help="JSONL produced by `extract`")
    p_upload.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=(
            "Corpus name. Sets the target Vectorize index and D1 database to "
            "this value (they must already exist; see README). Overrides "
            "VECTORIZE_INDEX and D1_DATABASE env vars. Default: mcpdf."
        ),
    )
    p_upload.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Cloudflare API calls; print what would happen.",
    )

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    if args.cmd == "extract":
        sys.exit(_cmd_extract(args))
    elif args.cmd == "upload":
        sys.exit(asyncio.run(_amain_upload(args)))
    else:  # pragma: no cover
        raise SystemExit(2)


if __name__ == "__main__":
    main()
