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


# D1's REST API caps per-query bound params at 100 and per-statement SQL at
# ~100KB. With 2 params per chunk row (vector_id, text), 40 chunks per batch
# = 80 params with comfortable headroom on both axes.
_CHUNKS_BATCH_SIZE = 40


def _vector_id(document_path: str, version: str, chunk_index: int) -> str:
    """Stable per-(path, version, chunk_index) vector ID.

    `version` is part of the document's identity: v15 and v16 of the same path
    occupy disjoint ID spaces and coexist in the index. Re-uploading the same
    (path, version) upserts in place. Pass version="" for unversioned uploads.
    The \\x00 separator prevents path/version boundary collisions
    (e.g. path="ab" v="c" vs path="abc" v="").
    """
    key = f"{document_path}\x00{version}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()[:16]
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

    input_root = args.path.expanduser().resolve()
    if args.absolute_paths:
        rel_root: Path | None = None
        log.info("Storing absolute document_path values")
    else:
        rel_root = input_root.parent if input_root.is_file() else input_root
        log.info("Storing document_path values relative to %s", rel_root)

    log.info("Extracting %d PDF(s) into %s", len(paths), args.out)
    seen_docs: set[str] = set()
    chunk_count = 0
    failures: list[ExtractFailure] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for item in extract_chunks(paths, tokenizer, cfg, path_relative_to=rel_root):
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


async def _insert_chunk_rows(
    client: CloudflareClient,
    db_id: str,
    rows: list[tuple[str, str]],
) -> None:
    """Multi-row UPSERT into `chunks`, batched to stay under D1's param cap."""
    for i in range(0, len(rows), _CHUNKS_BATCH_SIZE):
        batch = rows[i : i + _CHUNKS_BATCH_SIZE]
        placeholders = ",".join("(?, ?)" for _ in batch)
        params: list[object] = []
        for vector_id, text in batch:
            params.extend([vector_id, text])
        await client.d1_query(
            db_id,
            f"""
            INSERT INTO chunks (vector_id, text)
            VALUES {placeholders}
            ON CONFLICT(vector_id) DO UPDATE SET text = excluded.text
            """,
            params,
        )


async def _upload_one_doc(
    client: CloudflareClient,
    cfg: Config,
    chunks: list[ChunkRecord],
    db_id: str,
    *,
    dry_run: bool,
    version: str,
) -> None:
    doc = chunks[0]
    new_count = len(chunks)
    log.info(
        "%s: %d chunks (%s, %d pages)%s",
        doc.document_title,
        new_count,
        doc.document_path,
        doc.document_total_pages,
        f" [version={version}]" if version else "",
    )
    if dry_run:
        return

    # Look up previous chunk_count for this (path, version) so we can clean up
    # ghost vectors+rows after a shrinkage re-upload.
    prev = await client.d1_query(
        db_id,
        "SELECT chunk_count FROM documents WHERE path = ? AND version = ?",
        [doc.document_path, version],
    )
    old_count = int(prev[0]["chunk_count"]) if prev else 0

    # Embed in batches; build vector records (no text in metadata — that lives
    # in the chunks table) and parallel chunk rows.
    vectors: list[dict] = []
    chunk_rows: list[tuple[str, str]] = []
    for start in range(0, new_count, cfg.embed_batch_size):
        batch = chunks[start : start + cfg.embed_batch_size]
        embeddings = await client.embed_batch(cfg.workers_ai_model, [c.text for c in batch])
        for c, vec in zip(batch, embeddings):
            vid = _vector_id(c.document_path, version, c.chunk_index)
            vectors.append(
                {
                    "id": vid,
                    "values": vec,
                    "metadata": {
                        "document_path": c.document_path,
                        "document_title": c.document_title,
                        "document_sha256": c.document_sha256,
                        "version": version,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                    },
                }
            )
            chunk_rows.append((vid, c.text))
    await client.vectorize_insert(cfg.vectorize_index, vectors)
    await _insert_chunk_rows(client, db_id, chunk_rows)

    # Upsert the documents row. Composite PK (path, version) lets multiple
    # versions of the same path coexist; re-uploading the same (path, version)
    # upserts in place.
    await client.d1_query(
        db_id,
        """
        INSERT INTO documents (path, version, title, content_sha256, total_pages, chunk_count, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path, version) DO UPDATE SET
            title = excluded.title,
            content_sha256 = excluded.content_sha256,
            total_pages = excluded.total_pages,
            chunk_count = excluded.chunk_count,
            indexed_at = excluded.indexed_at
        """,
        [
            doc.document_path,
            version,
            doc.document_title,
            doc.document_sha256,
            doc.document_total_pages,
            new_count,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ],
    )

    # Shrinkage cleanup: if this (path, version) used to have more chunks,
    # the high-index IDs are now ghosts in both Vectorize and chunks.
    if old_count > new_count:
        ghost_ids = [
            _vector_id(doc.document_path, version, i) for i in range(new_count, old_count)
        ]
        log.info(
            "Cleaning up %d ghost chunks from previous upload (was %d, now %d)",
            len(ghost_ids),
            old_count,
            new_count,
        )
        await client.vectorize_delete_by_ids(cfg.vectorize_index, ghost_ids)
        # DELETE … WHERE id IN (?, ?, …) in batches under the 100-param cap.
        for i in range(0, len(ghost_ids), 90):
            batch_ids = ghost_ids[i : i + 90]
            placeholders = ",".join("?" for _ in batch_ids)
            await client.d1_query(
                db_id,
                f"DELETE FROM chunks WHERE vector_id IN ({placeholders})",
                list(batch_ids),
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
        version = args.version or ""
        for path, chunks in groups.items():
            try:
                await _upload_one_doc(
                    client, cfg, chunks, db_id, dry_run=args.dry_run, version=version
                )
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
    p_extract.add_argument(
        "--absolute-paths",
        action="store_true",
        help=(
            "Store the full absolute filesystem path in each chunk's "
            "document_path field. By default, paths are stored relative to "
            "the input root (or just the basename if the input is a single "
            "file). Don't mix relative and absolute paths in the same corpus — "
            "they're treated as different documents."
        ),
    )

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
        "--version",
        type=str,
        default=None,
        help=(
            "Optional version string. Part of the document's identity: "
            "uploading the same path with different --version values lets both "
            "coexist in the corpus (e.g. cubase_op_man.pdf v15 and v16). "
            "Re-uploading the same (path, version) upserts in place. "
            "Free-form string (e.g. '15', 'v1.2', '2026-Q2'). Omit to upload "
            "as the 'unversioned' entry (which is itself a distinct version)."
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
