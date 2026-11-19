"""Load files from DOCS_DIR, chunk them, embed them, and write them to the store.

Usage:
    python -m app.ingest            # uses DOCS_DIR (default data/docs)
    python -m app.ingest --reset    # drop the collection first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app import llm
from app.config import ConfigError, Settings, load_settings
from app.vectorstore import Chunk, VectorStore, open_store

SUPPORTED_SUFFIXES = (".txt", ".md")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character chunks with overlap, preferring whitespace boundaries."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def load_chunks(docs_dir: Path, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc_id = path.relative_to(docs_dir).as_posix()
        for i, piece in enumerate(chunk_text(path.read_text(encoding="utf-8"), size, overlap)):
            chunks.append(Chunk(chunk_id=f"{doc_id}::{i}", doc_id=doc_id, text=piece))
    return chunks


def ingest(settings: Settings, store: VectorStore, docs_dir: Path | None = None) -> int:
    docs_dir = Path(docs_dir or settings.docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"docs folder not found: {docs_dir}")
    chunks = load_chunks(docs_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise FileNotFoundError(f"no {'/'.join(SUPPORTED_SUFFIXES)} files with text found in {docs_dir}")
    store.upsert(chunks)
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--docs", help="override DOCS_DIR")
    parser.add_argument("--reset", action="store_true", help="drop the collection before ingesting")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    llm.configure(settings)
    store = open_store(settings)
    if args.reset:
        store.reset()
    try:
        n = ingest(settings, store, Path(args.docs) if args.docs else None)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ingested {n} chunks into collection {settings.collection_name!r} ({store.count()} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
