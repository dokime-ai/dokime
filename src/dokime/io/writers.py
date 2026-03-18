"""Dataset writers for JSONL, Parquet, and HuggingFace Hub output."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(data: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write an iterable of dicts to a JSONL file.

    Returns:
        Number of rows written.
    """
    count = 0
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_parquet(data: Iterable[dict[str, Any]], path: str | Path, batch_size: int = 10_000) -> int:
    """Write an iterable of dicts to a Parquet file.

    Returns:
        Number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    count = 0
    batch: list[dict[str, Any]] = []

    for row in data:
        batch.append(row)
        count += 1

        if len(batch) >= batch_size:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(str(path), table.schema)
            writer.write_table(table)
            batch = []

    if batch:
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(str(path), table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    return count


class StreamingWriter:
    """Buffered writer that flushes to disk in batches for constant memory usage.

    Supports JSONL and Parquet output. Format is inferred from file extension.

    Example::

        writer = StreamingWriter("output.parquet", batch_size=5000)
        for doc in pipeline.process(data):
            writer.write(doc)
        writer.close()
    """

    def __init__(self, path: str | Path, batch_size: int = 10_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.format = "parquet" if self.path.suffix == ".parquet" else "jsonl"
        self._buffer: list[dict[str, Any]] = []
        self._count = 0
        self._pq_writer: Any = None
        self._jsonl_file: Any = None

        if self.format == "jsonl":
            self._jsonl_file = Path(self.path).open("w", encoding="utf-8")  # noqa: SIM115

    def write(self, doc: dict[str, Any]) -> None:
        """Write a single document. Flushes to disk when buffer is full."""
        self._buffer.append(doc)
        self._count += 1

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        """Flush the buffer to disk."""
        if not self._buffer:
            return

        if self.format == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(self._buffer)
            if self._pq_writer is None:
                self._pq_writer = pq.ParquetWriter(str(self.path), table.schema)
            self._pq_writer.write_table(table)

        elif self.format == "jsonl" and self._jsonl_file is not None:
            for doc in self._buffer:
                self._jsonl_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
            self._jsonl_file.flush()

        self._buffer = []

    def close(self) -> None:
        """Flush remaining buffer and close file handles."""
        self._flush()
        if self._pq_writer is not None:
            self._pq_writer.close()
        if self._jsonl_file is not None:
            self._jsonl_file.close()

    @property
    def count(self) -> int:
        """Number of documents written so far."""
        return self._count

    def __enter__(self) -> StreamingWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
