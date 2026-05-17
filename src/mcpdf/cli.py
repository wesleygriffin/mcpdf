from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from . import indexer
from .config import Config


async def _amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mcpdf-index")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index a directory or single PDF")
    p_index.add_argument("path", type=Path)
    p_index.add_argument("--no-recursive", action="store_true")

    p_search = sub.add_parser("search", help="Run a one-shot search")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=8)

    args = parser.parse_args(argv)
    cfg = Config.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "index":
        result = await indexer.index_directory(cfg, args.path, recursive=not args.no_recursive)
        print(f"Indexed:     {len(result.indexed)}")
        print(f"Re-indexed:  {len(result.reindexed)}")
        print(f"Unchanged:   {len(result.skipped_unchanged)}")
        print(f"Failed:      {len(result.failed)}")
        for path, err in result.failed:
            print(f"  FAIL {path}: {err}")
        return 0

    if args.cmd == "search":
        hits = await indexer.search(cfg, args.query, top_k=args.top_k)
        if not hits:
            print("No results.")
            return 0
        for i, h in enumerate(hits, start=1):
            page = (
                f"p. {h.page_start}"
                if h.page_start == h.page_end
                else f"pp. {h.page_start}-{h.page_end}"
            )
            print(f"\n{i}. [{h.document_title}, {page}] (distance={h.distance:.3f})")
            print(f"   {h.document_path}")
            snippet = h.text.replace("\n", " ").strip()
            print(f"   {snippet[:400]}{'...' if len(snippet) > 400 else ''}")
        return 0

    return 1


def main() -> None:
    sys.exit(asyncio.run(_amain(sys.argv[1:])))


if __name__ == "__main__":
    main()
