"""Tests for streaming pipeline execution."""

import json
import tempfile
from pathlib import Path

from dokime.core.filters import LengthFilter
from dokime.core.pipeline import Pipeline
from dokime.io.writers import StreamingWriter

FIXTURES = Path(__file__).parent / "fixtures"


class TestStreamingWriter:
    def test_jsonl_streaming(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name

        writer = StreamingWriter(out_path, batch_size=2)
        for i in range(5):
            writer.write({"text": f"Document {i}", "id": i})
        writer.close()

        lines = Path(out_path).read_text().strip().split("\n")
        assert len(lines) == 5
        assert json.loads(lines[0])["id"] == 0
        Path(out_path).unlink()

    def test_parquet_streaming(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            out_path = f.name

        writer = StreamingWriter(out_path, batch_size=2)
        for i in range(5):
            writer.write({"text": f"Document {i}", "id": i})
        writer.close()

        import pyarrow.parquet as pq

        table = pq.read_table(out_path)
        assert len(table) == 5
        Path(out_path).unlink()

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name

        with StreamingWriter(out_path) as writer:
            writer.write({"text": "hello"})
            writer.write({"text": "world"})

        assert writer.count == 2
        Path(out_path).unlink()


class TestStreamingPipeline:
    def test_pipeline_uses_streaming(self):
        """Pipeline.run should use StreamingWriter, not buffer everything."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name

        pipeline = Pipeline("stream-test").add_filter(LengthFilter(min_length=20))
        result = pipeline.run(str(FIXTURES / "sample.jsonl"), out_path, quiet=True)

        assert result["total_read"] == 10
        assert result["total_kept"] > 0

        lines = Path(out_path).read_text().strip().split("\n")
        assert len(lines) == result["total_kept"]
        Path(out_path).unlink()

    def test_process_iterator(self):
        """Pipeline.process should yield documents without writing."""
        pipeline = Pipeline("iter-test").add_filter(LengthFilter(min_length=20))
        kept = list(pipeline.process(str(FIXTURES / "sample.jsonl")))
        assert len(kept) > 0
        assert all(len(doc["text"]) >= 20 for doc in kept)
