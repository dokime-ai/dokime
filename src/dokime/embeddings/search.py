"""Semantic search and nearest-neighbor operations on embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    """A single search result with document, score, and index."""

    document: dict[str, Any]
    score: float
    index: int


class EmbeddingIndex:
    """FAISS-based index for fast nearest-neighbor search over embeddings.

    Example::

        index = EmbeddingIndex(embeddings, documents)
        results = index.search("machine learning", model, k=5)
        for r in results:
            print(f"{r.score:.3f}: {r.document['text'][:80]}")
    """

    def __init__(self, embeddings: np.ndarray, documents: list[dict[str, Any]]) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install embedding support: pip install dokime[embeddings]") from None

        self.documents = documents
        self.embeddings = embeddings.astype(np.float32)
        self.dimension = embeddings.shape[1]

        # Build FAISS index (inner product on normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

    def search(
        self,
        query: str,
        model: Any,
        k: int = 10,
    ) -> list[SearchResult]:
        """Search for documents similar to a text query.

        Args:
            query: Text query to search for.
            model: EmbeddingModel instance for encoding the query.
            k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by similarity (highest first).
        """
        query_embedding = model.encode_texts([query], show_progress=False)
        return self.search_by_vector(query_embedding[0], k=k)

    def search_by_vector(self, vector: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Search for documents similar to a given embedding vector."""
        query = vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, min(k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append(SearchResult(document=self.documents[idx], score=float(score), index=int(idx)))

        return results

    def find_neighbors(self, idx: int, k: int = 10) -> list[SearchResult]:
        """Find the k nearest neighbors of a document by its index."""
        return self.search_by_vector(self.embeddings[idx], k=k + 1)[1:]  # exclude self


class AnomalyScorer:
    """Score documents by how anomalous they are, using k-NN distance in embedding space.

    Low k-NN distance = typical sample (surrounded by similar documents).
    High k-NN distance = anomalous/outlier (isolated in embedding space).

    Example::

        scorer = AnomalyScorer(embeddings)
        scores = scorer.score_all(k=10)
        # scores[i] = average distance to 10 nearest neighbors for document i
        outlier_indices = np.argsort(scores)[-100:]  # top 100 outliers
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install embedding support: pip install dokime[embeddings]") from None

        self.embeddings = embeddings.astype(np.float32)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

    def score_all(self, k: int = 10) -> np.ndarray:
        """Compute anomaly scores for all documents.

        Returns:
            Array of shape (n_docs,) with anomaly scores.
            Higher score = more anomalous (lower average similarity to neighbors).
        """
        k_actual = min(k + 1, len(self.embeddings))
        scores, _ = self.index.search(self.embeddings, k_actual)

        # Average similarity to k nearest neighbors (excluding self)
        # Lower similarity = higher anomaly
        avg_similarity = scores[:, 1:].mean(axis=1)

        # Invert: higher anomaly score = more unusual
        anomaly_scores = 1.0 - avg_similarity

        return np.asarray(anomaly_scores)

    def find_outliers(self, k: int = 10, threshold: float | None = None, top_n: int | None = None) -> list[int]:
        """Find outlier document indices.

        Args:
            k: Number of neighbors for scoring.
            threshold: Anomaly score threshold. Documents above this are outliers.
            top_n: Return the top N most anomalous documents.

        Returns:
            List of document indices identified as outliers.
        """
        scores = self.score_all(k=k)

        if top_n is not None:
            indices = np.argsort(scores)[-top_n:]
            return sorted(indices.tolist(), key=lambda i: scores[i], reverse=True)

        if threshold is not None:
            return [i for i, s in enumerate(scores) if s > threshold]

        # Default: flag documents > 2 standard deviations above mean
        mean = scores.mean()
        std = scores.std()
        return [i for i, s in enumerate(scores) if s > mean + 2 * std]
