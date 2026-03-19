"""Semantic deduplication using embeddings — finds near-duplicates that hash-based methods miss."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console

console = Console()


def find_semantic_duplicates(
    embeddings: np.ndarray,
    documents: list[dict[str, Any]],
    threshold: float = 0.95,
    quiet: bool = False,
) -> list[tuple[int, int, float]]:
    """Find pairs of semantically similar documents using embedding cosine similarity.

    Args:
        embeddings: Array of shape (n_docs, embed_dim), L2-normalized.
        documents: List of document dicts (for reference).
        threshold: Cosine similarity threshold to consider as duplicate (0-1).
        quiet: Suppress output.

    Returns:
        List of (idx_a, idx_b, similarity) tuples for duplicate pairs.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Install embedding support: pip install dokime[embeddings]") from None

    n = len(embeddings)
    emb = embeddings.astype(np.float32)

    if not quiet:
        console.print(f"[bold blue]Dokime[/] — Semantic dedup on {n:,} documents (threshold={threshold})")

    # Build index
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # For each document, find neighbors above threshold
    # Search for k neighbors — we don't know how many dupes exist, so start with k=10
    k = min(10, n)
    scores, indices = index.search(emb, k)

    duplicate_pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for i in range(n):
        for j_pos in range(1, k):  # skip self (position 0)
            j = int(indices[i, j_pos])
            sim = float(scores[i, j_pos])

            if sim < threshold:
                break

            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                duplicate_pairs.append((pair[0], pair[1], sim))

    if not quiet:
        console.print(f"  Found {len(duplicate_pairs):,} duplicate pairs above threshold {threshold}")

    return duplicate_pairs


def deduplicate_by_embeddings(
    embeddings: np.ndarray,
    documents: list[dict[str, Any]],
    threshold: float = 0.95,
    quiet: bool = False,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Remove semantic duplicates, keeping the first occurrence.

    Args:
        embeddings: Array of shape (n_docs, embed_dim), L2-normalized.
        documents: List of document dicts.
        threshold: Cosine similarity threshold.
        quiet: Suppress output.

    Returns:
        Tuple of (deduplicated documents, indices of kept documents).
    """
    pairs = find_semantic_duplicates(embeddings, documents, threshold, quiet=quiet)

    # Build set of indices to remove (keep the lower index in each pair)
    to_remove: set[int] = set()
    for _a, b, _sim in pairs:
        to_remove.add(b)

    kept_indices = [i for i in range(len(documents)) if i not in to_remove]
    kept_docs = [documents[i] for i in kept_indices]

    if not quiet:
        console.print(f"  Kept {len(kept_docs):,} / {len(documents):,} documents")
        console.print(f"  Removed {len(to_remove):,} semantic duplicates")

    return kept_docs, kept_indices
