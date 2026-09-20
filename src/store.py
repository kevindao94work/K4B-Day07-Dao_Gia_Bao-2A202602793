from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Uses an in-memory store with one record per supplied Document.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = deepcopy(doc.metadata)
        parent_id, separator, chunk_index = doc.id.rpartition("#")
        fallback_id = parent_id if parent_id and separator and chunk_index.isdigit() else doc.id
        # Explicit source identity takes priority over the chunk naming convention.
        if not metadata.get("doc_id"):
            metadata["doc_id"] = fallback_id
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": list(self._embedding_fn(doc.content)),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []
        query_embedding = self._embedding_fn(query)
        ranked = sorted(
            ((_dot(query_embedding, record["embedding"]), record) for record in records),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            {"id": record["id"], "content": record["content"],
             "metadata": deepcopy(record["metadata"]), "score": score}
            for score, record in ranked[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        Append one record per document; chunking is the caller's responsibility.
        """
        records = [self._make_record(doc) for doc in docs]
        self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        candidates = self._store
        if metadata_filter:
            candidates = [
                record for record in self._store
                if all(key in record["metadata"] and record["metadata"][key] == value
                       for key, value in metadata_filter.items())
            ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        previous_size = len(self._store)
        self._store = [record for record in self._store
                       if record["metadata"]["doc_id"] != doc_id]
        return len(self._store) < previous_size
