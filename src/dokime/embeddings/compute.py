"""Compute embeddings for text documents using sentence-transformers or custom models."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from tqdm import tqdm

console = Console()


class EmbeddingModel:
    """Wrapper around sentence-transformers for computing text embeddings.

    Example::

        model = EmbeddingModel("all-MiniLM-L6-v2")
        embeddings = model.encode_texts(["hello world", "foo bar"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install embedding support: pip install dokime-ai[embeddings]") from None

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )


def compute_embeddings(
    data: Iterator[dict[str, Any]] | list[dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    text_field: str = "text",
    batch_size: int = 64,
    output_path: str | Path | None = None,
    device: str | None = None,
    quiet: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Compute embeddings for a dataset.

    Args:
        data: Iterator or list of document dicts.
        model_name: Name of the sentence-transformer model.
        text_field: Field containing the text to embed.
        batch_size: Batch size for encoding.
        output_path: Optional path to save embeddings as .npy file.
        device: Device for inference ('cpu', 'cuda', etc.).
        quiet: Suppress progress output.

    Returns:
        Tuple of (documents list, embeddings array of shape [n_docs, embed_dim]).
    """
    if not quiet:
        console.print(f"[bold blue]Dokime[/] — Computing embeddings with [bold]{model_name}[/]")

    model = EmbeddingModel(model_name, device=device)

    # Collect all documents and texts
    documents: list[dict[str, Any]] = []
    texts: list[str] = []

    for sample in tqdm(data, desc="Loading", unit=" docs", disable=quiet):
        documents.append(sample)
        texts.append(sample.get(text_field, ""))

    if not quiet:
        console.print(f"  Loaded {len(documents):,} documents")
        console.print(f"  Model: {model_name} ({model.dimension}d)")

    # Compute embeddings
    embeddings = model.encode_texts(texts, batch_size=batch_size, show_progress=not quiet)

    if not quiet:
        console.print(f"  Embeddings shape: {embeddings.shape}")

    # Optionally save
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), embeddings)
        if not quiet:
            console.print(f"  Saved to: {out}")

    return documents, embeddings
