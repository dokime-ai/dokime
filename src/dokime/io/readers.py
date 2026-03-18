"""Dataset readers for JSONL, Parquet, CSV, and HuggingFace datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file, yielding one document dict per line."""
    import json

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_parquet(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a Parquet file, yielding one document dict per row."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    for batch in table.to_batches():
        yield from batch.to_pylist()


def read_csv(path: str | Path, text_field: str = "text") -> Iterator[dict[str, Any]]:
    """Read a CSV file, yielding one document dict per row."""
    import csv

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def read_dataset(path: str, split: str = "train") -> Iterator[dict[str, Any]]:
    """Read a HuggingFace dataset, yielding one document dict per example.

    Requires: pip install dokime-ai[io]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets support: pip install dokime-ai[io]") from None

    ds = load_dataset(path, split=split, streaming=True)
    yield from ds


def auto_read(path: str) -> Iterator[dict[str, Any]]:
    """Automatically detect format and read a dataset.

    Supports: .jsonl, .parquet, .csv, and HuggingFace dataset IDs.
    """
    p = Path(path)

    if p.suffix == ".jsonl" or p.suffix == ".jsonlines":
        yield from read_jsonl(p)
    elif p.suffix == ".parquet":
        yield from read_parquet(p)
    elif p.suffix == ".csv":
        yield from read_csv(p)
    elif not p.exists():
        # Assume it's a HuggingFace dataset ID
        yield from read_dataset(path)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Supported: .jsonl, .parquet, .csv")
