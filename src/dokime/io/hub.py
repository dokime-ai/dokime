"""HuggingFace Hub integration — load and push datasets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from rich.console import Console

console = Console()


def load_from_hub(
    dataset_id: str,
    split: str = "train",
    streaming: bool = True,
    text_field: str = "text",
) -> Iterator[dict[str, Any]]:
    """Load a dataset from HuggingFace Hub.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "allenai/c4", "HuggingFaceFW/fineweb").
        split: Dataset split to load.
        streaming: Stream the dataset (recommended for large datasets).
        text_field: Not used directly, but indicates the expected text field.

    Yields:
        Document dicts from the dataset.

    Requires: pip install dokime-ai[io]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace support: pip install dokime-ai[io]") from None

    console.print(f"[bold blue]Dokime[/] — Loading [bold]{dataset_id}[/] (split={split}, streaming={streaming})")

    ds = load_dataset(dataset_id, split=split, streaming=streaming)
    yield from ds


def push_to_hub(
    data: Iterator[dict[str, Any]] | list[dict[str, Any]],
    repo_id: str,
    split: str = "train",
    private: bool = False,
    token: str | None = None,
    max_shard_size: str = "500MB",
) -> str:
    """Push a curated dataset to HuggingFace Hub.

    Args:
        data: Iterator or list of document dicts.
        repo_id: Target repo (e.g., "your-username/my-curated-dataset").
        split: Split name for the upload.
        private: Whether the dataset should be private.
        token: HuggingFace API token. If None, uses cached login.
        max_shard_size: Max shard size for upload.

    Returns:
        URL of the uploaded dataset.

    Requires: pip install dokime-ai[io]
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Install HuggingFace support: pip install dokime-ai[io]") from None

    console.print(f"[bold blue]Dokime[/] — Pushing to [bold]{repo_id}[/]")

    if not isinstance(data, list):
        data = list(data)

    ds = Dataset.from_list(data)

    ds.push_to_hub(
        repo_id,
        split=split,
        private=private,
        token=token,
        max_shard_size=max_shard_size,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    console.print(f"  Pushed {len(data):,} documents to [link={url}]{url}[/link]")
    return url
