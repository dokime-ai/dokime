"""Pipeline orchestration for data curation workflows."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from dokime.core.filters import Filter
from dokime.core.registry import FILTER_REGISTRY

logger = logging.getLogger(__name__)
console = Console()


class Pipeline:
    """A composable data curation pipeline with streaming execution.

    Processes documents one at a time through the filter chain, never loading
    the full dataset into memory. Supports writing output in configurable
    batch sizes for constant memory usage regardless of dataset size.

    Example::

        pipeline = Pipeline("my-pipeline")
        pipeline.add_filter(LengthFilter(min_length=100))
        pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
        pipeline.add_filter(MinHashDedup(threshold=0.8))
        result = pipeline.run("data/raw.jsonl", "data/curated.parquet")
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.filters: list[Filter] = []

    def add_filter(self, f: Filter) -> Pipeline:
        """Add a filter to the pipeline. Returns self for chaining."""
        self.filters.append(f)
        return self

    @classmethod
    def from_config(cls, config_path: str | Path) -> Pipeline:
        """Load a pipeline from a YAML config file.

        Example YAML::

            name: my-pipeline
            output_format: parquet          # parquet or jsonl (default: inferred from path)
            write_batch_size: 10000         # flush to disk every N docs (default: 10000)

            filters:
              - LengthFilter:
                  min_length: 100
                  max_length: 100000
              - WhitespaceFilter:
                  max_whitespace_ratio: 0.4
              - RepetitionFilter:
                  max_repetition_ratio: 0.3
              - ExactDedup: {}
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        pipeline = cls(name=config.get("name", "default"))
        pipeline._config = config

        for filter_spec in config.get("filters", []):
            if isinstance(filter_spec, str):
                filter_cls = FILTER_REGISTRY[filter_spec]
                pipeline.add_filter(filter_cls())
            elif isinstance(filter_spec, dict):
                for filter_name, filter_args in filter_spec.items():
                    filter_cls = FILTER_REGISTRY[filter_name]
                    if filter_args:
                        pipeline.add_filter(filter_cls(**filter_args))
                    else:
                        pipeline.add_filter(filter_cls())

        return pipeline

    def process(self, data: Iterator[dict[str, Any]] | str) -> Iterator[dict[str, Any]]:
        """Stream documents through the filter chain, yielding kept documents.

        This is the core streaming API — never buffers the full dataset.
        Documents that raise exceptions during filtering are skipped with a warning.

        Args:
            data: Iterator of document dicts, or a file path string.

        Yields:
            Documents that pass all filters.
        """
        if isinstance(data, str):
            from dokime.io.readers import auto_read

            data = auto_read(data)

        for doc_idx, sample in enumerate(data):
            try:
                kept = True
                for f in self.filters:
                    if not f.filter(sample):
                        kept = False
                        break
                if kept:
                    yield sample
            except Exception:
                logger.warning("Skipping document %d due to filter error", doc_idx, exc_info=True)

    def run(
        self,
        input_path: str,
        output_path: str | None = None,
        quiet: bool = False,
        write_batch_size: int = 10_000,
    ) -> dict[str, Any]:
        """Execute the pipeline with streaming I/O.

        Writes output in batches for constant memory usage. Never loads
        the full dataset into memory.

        Args:
            input_path: Path to input dataset (JSONL, Parquet, CSV, or HuggingFace dataset ID).
            output_path: Path for curated output. Format inferred from extension.
            quiet: Suppress progress output.
            write_batch_size: Flush to disk every N kept documents. Lower = less memory.

        Returns:
            Dictionary with pipeline execution statistics.
        """
        from tqdm import tqdm

        from dokime.io.readers import auto_read
        from dokime.io.writers import StreamingWriter

        start_time = time.time()

        filter_stats: dict[str, int] = {f.name(): 0 for f in self.filters}
        total_read = 0
        total_kept = 0

        if not quiet:
            console.print(f"[bold blue]Dokime[/] — Running pipeline [bold]{self.name}[/]")
            console.print(f"  Input:   {input_path}")
            console.print(f"  Output:  {output_path or '(dry run)'}")
            console.print(f"  Filters: {len(self.filters)}")
            console.print()

        # Set up streaming writer
        writer = StreamingWriter(output_path, batch_size=write_batch_size) if output_path else None

        reader = auto_read(input_path)
        progress = tqdm(reader, desc="Processing", unit=" docs", disable=quiet)

        error_count = 0
        for sample in progress:
            total_read += 1

            try:
                kept = True
                for f in self.filters:
                    if not f.filter(sample):
                        filter_stats[f.name()] += 1
                        kept = False
                        break

                if kept:
                    total_kept += 1
                    if writer is not None:
                        writer.write(sample)
            except Exception:
                error_count += 1
                logger.warning("Skipping document %d due to filter error", total_read, exc_info=True)

        progress.close()

        if error_count > 0 and not quiet:
            console.print(f"  [yellow]WARNING: {error_count:,} documents skipped due to errors[/]")

        if writer is not None:
            writer.close()

        elapsed = time.time() - start_time

        total_removed = total_read - total_kept
        removal_rate = (total_removed / total_read * 100) if total_read > 0 else 0.0
        docs_per_sec = total_read / elapsed if elapsed > 0 else 0.0

        stats: dict[str, Any] = {
            "pipeline": self.name,
            "input": input_path,
            "output": output_path,
            "total_read": total_read,
            "total_kept": total_kept,
            "total_removed": total_removed,
            "errors_skipped": error_count,
            "removal_rate_pct": round(removal_rate, 2),
            "filters_applied": len(self.filters),
            "per_filter_removed": filter_stats,
            "elapsed_seconds": round(elapsed, 2),
            "docs_per_second": round(docs_per_sec, 1),
        }

        if not quiet:
            self._print_stats(stats)

        return stats

    def _print_stats(self, stats: dict[str, Any]) -> None:
        """Print a summary table of pipeline results."""
        console.print()

        table = Table(title="Pipeline Results", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Documents read", f"{stats['total_read']:,}")
        table.add_row("Documents kept", f"[green]{stats['total_kept']:,}[/]")
        table.add_row("Documents removed", f"[red]{stats['total_removed']:,}[/]")
        table.add_row("Removal rate", f"{stats['removal_rate_pct']}%")
        table.add_row("Elapsed", f"{stats['elapsed_seconds']}s")
        table.add_row("Throughput", f"{stats['docs_per_second']:,.0f} docs/s")

        console.print(table)

        if stats["per_filter_removed"]:
            console.print()
            ft = Table(title="Per-Filter Removals", show_header=True)
            ft.add_column("Filter", style="bold")
            ft.add_column("Removed", justify="right")

            for fname, count in stats["per_filter_removed"].items():
                ft.add_row(fname, f"{count:,}")

            console.print(ft)
