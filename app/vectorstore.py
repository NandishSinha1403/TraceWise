"""Vector store interface and the Chroma implementation used in M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app import llm
from app.state import Passage


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[Passage]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class ChromaStore:
    """Chroma collection. Embeddings are always computed via `app.llm.embed`."""

    def __init__(self, collection_name: str, persist_dir: str | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(settings=chroma_settings)
        else:
            self._client = chromadb.PersistentClient(path=persist_dir, settings=chroma_settings)
        self._name = collection_name
        self._collection = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = llm.embed([c.text for c in chunks], task="document")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id} for c in chunks],
        )

    def search(self, query: str, k: int) -> list[Passage]:
        n = min(k, self.count())
        if n == 0:
            return []
        [vector] = llm.embed([query], task="query")
        res = self._collection.query(query_embeddings=[vector], n_results=n)
        return [
            Passage(chunk_id=cid, doc_id=meta["doc_id"], text=doc, score=1.0 - float(dist))
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._open()


def open_store(settings) -> VectorStore:
    if settings.vector_store == "chroma":
        return ChromaStore(settings.collection_name, persist_dir=settings.chroma_dir)
    raise ValueError(f"unsupported VECTOR_STORE {settings.vector_store!r}")
