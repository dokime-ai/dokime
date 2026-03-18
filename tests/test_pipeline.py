"""Tests for pipeline execution and YAML config loading."""

import json
import tempfile
from pathlib import Path

from dokime.core.filters import LengthFilter, WhitespaceFilter
from dokime.core.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"


class TestPipelineRun:
    def test_run_with_length_filter(self):
        pipeline = Pipeline("test")
        pipeline.add_filter(LengthFilter(min_length=20))

        result = pipeline.run(str(FIXTURES / "sample.jsonl"), quiet=True)

        assert result["total_read"] == 10
        assert result["total_kept"] < 10  # Some short docs removed
        assert result["total_removed"] > 0
        assert result["removal_rate_pct"] > 0

    def test_run_with_output_jsonl(self):
        pipeline = Pipeline("test")
        pipeline.add_filter(LengthFilter(min_length=20))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name

        result = pipeline.run(str(FIXTURES / "sample.jsonl"), out_path, quiet=True)

        # Verify output file was written
        output_lines = Path(out_path).read_text().strip().split("\n")
        assert len(output_lines) == result["total_kept"]

        # Verify each line is valid JSON
        for line in output_lines:
            doc = json.loads(line)
            assert "text" in doc
            assert len(doc["text"]) >= 20

        Path(out_path).unlink()

    def test_run_with_output_parquet(self):
        pipeline = Pipeline("test")
        pipeline.add_filter(LengthFilter(min_length=20))

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            out_path = f.name

        result = pipeline.run(str(FIXTURES / "sample.jsonl"), out_path, quiet=True)

        import pyarrow.parquet as pq

        table = pq.read_table(out_path)
        assert len(table) == result["total_kept"]

        Path(out_path).unlink()

    def test_run_chained_filters(self):
        pipeline = Pipeline("test")
        pipeline.add_filter(LengthFilter(min_length=20))
        pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.5))

        result = pipeline.run(str(FIXTURES / "sample.jsonl"), quiet=True)

        # Both filters should have removed something
        assert result["total_removed"] > 0
        assert len(result["per_filter_removed"]) == 2

    def test_run_dry_run(self):
        """Pipeline with no output_path returns stats without writing."""
        pipeline = Pipeline("test")
        pipeline.add_filter(LengthFilter(min_length=20))

        result = pipeline.run(str(FIXTURES / "sample.jsonl"), quiet=True)

        assert result["output"] is None
        assert result["total_read"] == 10

    def test_empty_pipeline(self):
        pipeline = Pipeline("empty")
        result = pipeline.run(str(FIXTURES / "sample.jsonl"), quiet=True)

        assert result["total_read"] == 10
        assert result["total_kept"] == 10
        assert result["total_removed"] == 0

    def test_chaining_api(self):
        pipeline = (
            Pipeline("chained")
            .add_filter(LengthFilter(min_length=20))
            .add_filter(WhitespaceFilter(max_whitespace_ratio=0.5))
        )
        assert len(pipeline.filters) == 2


class TestPipelineConfig:
    def test_load_from_yaml(self):
        pipeline = Pipeline.from_config(str(FIXTURES / ".." / ".." / "examples" / "basic_pipeline.yaml"))

        assert pipeline.name == "basic-quality-filters"
        assert len(pipeline.filters) == 5

    def test_yaml_pipeline_runs(self):
        pipeline = Pipeline.from_config(str(FIXTURES / ".." / ".." / "examples" / "basic_pipeline.yaml"))

        result = pipeline.run(str(FIXTURES / "sample.jsonl"), quiet=True)

        assert result["total_read"] == 10
        assert result["total_kept"] < 10
        assert result["total_removed"] > 0
