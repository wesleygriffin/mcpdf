from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sqlite_vec


@dataclass(frozen=True)
class SearchHit:
    document_id: int
    document_title: str
    document_path: str
    chunk_id: int
    page_start: int
    page_end: int
    text: str
    distance: float  # smaller = closer


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY,
    path            TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    total_pages     INTEGER NOT NULL,
    indexed_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    page_start    INTEGER NOT NULL,
    page_end      INTEGER NOT NULL,
    token_count   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


def connect(db_path: Path, *, embedding_dim: int) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.executescript(_SCHEMA)
    # vec0 virtual table — must be created with a fixed dim per DB.
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors "
        f"USING vec0(embedding FLOAT[{embedding_dim}])"
    )
    return conn


def get_document_by_path(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()


def delete_document(conn: sqlite3.Connection, document_id: int) -> None:
    # Find chunk ids first so we can scrub the vec0 rows (no FK cascades into vec0).
    chunk_ids = [
        row["id"]
        for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
    ]
    with conn:
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            conn.execute(f"DELETE FROM chunk_vectors WHERE rowid IN ({placeholders})", chunk_ids)
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))


def insert_document(
    conn: sqlite3.Connection,
    *,
    path: str,
    title: str,
    content_sha256: str,
    total_pages: int,
) -> int:
    cur = conn.execute(
        "INSERT INTO documents (path, title, content_sha256, total_pages) VALUES (?, ?, ?, ?)",
        (path, title, content_sha256, total_pages),
    )
    return int(cur.lastrowid)


def insert_chunks(
    conn: sqlite3.Connection,
    *,
    document_id: int,
    chunks_with_embeddings: Iterable[tuple[int, int, int, int, str, list[float]]],
) -> None:
    """Each tuple: (chunk_index, page_start, page_end, token_count, text, embedding)."""
    with conn:
        for chunk_index, page_start, page_end, token_count, text, embedding in chunks_with_embeddings:
            cur = conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, page_start, page_end, token_count, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (document_id, chunk_index, page_start, page_end, token_count, text),
            )
            chunk_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
                (chunk_id, json.dumps(embedding)),
            )


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT d.id, d.title, d.path, d.total_pages, d.indexed_at, "
            "       COUNT(c.id) AS chunk_count "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.id ORDER BY d.title"
        )
    )


def get_document(conn: sqlite3.Connection, document_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, title, path, total_pages, content_sha256, indexed_at "
        "FROM documents WHERE id = ?",
        (document_id,),
    ).fetchone()


def search(
    conn: sqlite3.Connection,
    *,
    query_embedding: list[float],
    top_k: int,
    document_ids: list[int] | None = None,
) -> list[SearchHit]:
    # vec0 KNN: order by `distance` after a MATCH on a JSON-encoded vector.
    # We pull a generous candidate window then filter, because vec0's MATCH
    # operates over the whole table — post-filtering by document_id is fine
    # for collections up to ~100k chunks.
    candidate_k = top_k * (5 if document_ids else 1)
    rows = conn.execute(
        """
        SELECT v.rowid AS chunk_id, v.distance AS distance,
               c.document_id, c.page_start, c.page_end, c.text,
               d.title AS document_title, d.path AS document_path
        FROM chunk_vectors v
        JOIN chunks    c ON c.id = v.rowid
        JOIN documents d ON d.id = c.document_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (json.dumps(query_embedding), candidate_k),
    ).fetchall()

    hits: list[SearchHit] = []
    for row in rows:
        if document_ids and row["document_id"] not in document_ids:
            continue
        hits.append(
            SearchHit(
                document_id=row["document_id"],
                document_title=row["document_title"],
                document_path=row["document_path"],
                chunk_id=row["chunk_id"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                text=row["text"],
                distance=row["distance"],
            )
        )
        if len(hits) >= top_k:
            break
    return hits
