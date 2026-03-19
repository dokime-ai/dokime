"""Tests for bug fixes in the pre-launch sprint.

Covers:
- Parquet reader streaming (iter_batches, not read_table)
- compute_embeddings missing field warning
- Pipeline per-document error handling
- CLI --field parameter consistency
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dokime.core.filters import Filter
from dokime.core.pipeline import Pipeline
from dokime.io.readers import read_parquet

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Parquet reader streaming
# ---------------------------------------------------------------------------


class TestParquetStreaming:
    def _write_parquet(self, rows: list[dict], path: str) -> None:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)

    def test_reads_all_rows(self):
        rows = [{"text": f"Document {i}", "id": i} for i in range(50)]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        self._write_parquet(rows, path)

        result = list(read_parquet(path))
        assert len(result) == 50
        assert result[0]["text"] == "Document 0"
        assert result[49]["id"] == 49
        Path(path).unlink()

    def test_batch_size_does_not_affect_result(self):
        """Different batch sizes should yield identical results."""
        rows = [{"text": f"Row {i}"} for i in range(25)]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        self._write_parquet(rows, path)

        result_small = list(read_parquet(path, batch_size=5))
        result_large = list(read_parquet(path, batch_size=100))
        assert result_small == result_large
        Path(path).unlink()

    def test_empty_parquet(self):
        """Empty Parquet file should yield no rows."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        table = pa.table({"text": pa.array([], type=pa.string())})
        pq.write_table(table, path)

        result = list(read_parquet(path))
        assert len(result) == 0
        Path(path).unlink()

    def test_uses_iter_batches_not_read_table(self):
        """Verify we use ParquetFile.iter_batches (streaming) not read_table (full load)."""
        rows = [{"text": "hello"}]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        self._write_parquet(rows, path)

        with patch("pyarrow.parquet.ParquetFile") as mock_pf_cls:
            mock_pf = MagicMock()
            mock_batch = MagicMock()
            mock_batch.to_pylist.return_value = [{"text": "hello"}]
            mock_pf.iter_batches.return_value = [mock_batch]
            mock_pf_cls.return_value = mock_pf

            result = list(read_parquet(path))

            mock_pf.iter_batches.assert_called_once()
            assert result == [{"text": "hello"}]
        Path(path).unlink()


# ---------------------------------------------------------------------------
# compute_embeddings missing field warning
# ---------------------------------------------------------------------------


class TestComputeEmbeddingsMissingField:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("numpy"),
        reason="numpy not installed",
    )
    def test_warns_on_missing_field(self, caplog):
        """compute_embeddings should log a warning when documents lack the text field."""
        from dokime.embeddings.compute import compute_embeddings

        docs = [
            {"content": "This uses 'content' not 'text'"},
            {"content": "Another doc without 'text' field"},
        ]

        with patch("dokime.embeddings.compute.EmbeddingModel") as mock_model_cls:
            import numpy as np

            mock_instance = MagicMock()
            mock_instance.dimension = 384
            mock_instance.model_name = "test"
            mock_instance.encode_texts.return_value = np.zeros((2, 384))
            mock_model_cls.return_value = mock_instance

            with caplog.at_level(logging.WARNING, logger="dokime.embeddings.compute"):
                documents, _embeddings = compute_embeddings(
                    data=iter(docs),
                    model_name="test-model",
                    text_field="text",
                    quiet=True,
                )

            assert "2 of 2 documents lack field 'text'" in caplog.text
            assert len(documents) == 2
            # Verify empty strings were passed for embedding
            call_args = mock_instance.encode_texts.call_args
            texts_passed = call_args[0][0]
            assert texts_passed == ["", ""]


# ---------------------------------------------------------------------------
# Pipeline per-document error handling
# ---------------------------------------------------------------------------


class _ExplodingFilter(Filter):
    """Filter that raises on specific documents, for testing error handling."""

    def __init__(self, explode_on: str = "EXPLODE"):
        self.explode_on = explode_on

    def filter(self, sample: dict) -> bool:
        if self.explode_on in sample.get("text", ""):
            raise ValueError(f"Intentional test error on: {sample['text']}")
        return True

    def name(self) -> str:
        return "ExplodingFilter"


class TestPipelineErrorHandling:
    def test_process_skips_erroring_documents(self):
        """Pipeline.process should skip documents that cause filter errors."""
        pipeline = Pipeline("error-test")
        pipeline.add_filter(_ExplodingFilter(explode_on="BOOM"))

        docs = [
            {"text": "Good document"},
            {"text": "BOOM goes the dynamite"},
            {"text": "Another good one"},
        ]

        result = list(pipeline.process(iter(docs)))
        assert len(result) == 2
        assert result[0]["text"] == "Good document"
        assert result[1]["text"] == "Another good one"

    def test_run_reports_error_count(self):
        """Pipeline.run should report errors_skipped in stats."""
        # Write test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for text in ["Good doc one", "KABOOM bad doc", "Good doc two"]:
                f.write(json.dumps({"text": text}) + "\n")
            path = f.name

        pipeline = Pipeline("error-test")
        pipeline.add_filter(_ExplodingFilter(explode_on="KABOOM"))

        result = pipeline.run(path, quiet=True)

        assert result["total_read"] == 3
        assert result["total_kept"] == 2
        assert result["errors_skipped"] == 1
        Path(path).unlink()

    def test_no_errors_means_zero_count(self):
        """When no errors occur, errors_skipped should be 0."""
        result = Pipeline("clean").run(str(FIXTURES / "sample.jsonl"), quiet=True)
        assert result["errors_skipped"] == 0


# ---------------------------------------------------------------------------
# Static file packaging for explore
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("fastapi"),
    reason="fastapi not installed",
)
class TestExploreStaticFiles:
    def test_static_dir_exists(self):
        """The explore/static directory must exist in the package."""
        from dokime.explore.server import STATIC_DIR

        assert STATIC_DIR.exists(), f"STATIC_DIR not found: {STATIC_DIR}"
        assert (STATIC_DIR / "index.html").exists(), "index.html not found in STATIC_DIR"

    def test_index_html_is_not_empty(self):
        """index.html must have content."""
        from dokime.explore.server import STATIC_DIR

        content = (STATIC_DIR / "index.html").read_text()
        assert len(content) > 100
        assert "<html" in content.lower()
