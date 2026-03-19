"""Dataset readers for JSONL, Parquet, CSV, and HuggingFace datasets."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file, yielding one document dict per line.

    Skips blank lines and malformed JSON lines with a warning.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d in %s", line_num, path)


def read_parquet(path: str | Path, batch_size: int = 10_000) -> Iterator[dict[str, Any]]:
    """Read a Parquet file, yielding one document dict per row.

    Uses iter_batches for constant memory usage — never loads the full file.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def read_csv(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a CSV file, yielding one document dict per row."""
    import csv

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def read_dataset(path: str, split: str = "train") -> Iterator[dict[str, Any]]:
    """Read a HuggingFace dataset, yielding one document dict per example.

    Requires: pip install dokime[io]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets support: pip install dokime[io]") from None

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
        yield from read_dataset(path)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Supported: .jsonl, .parquet, .csv")
