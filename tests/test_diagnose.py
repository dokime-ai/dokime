"""Tests for the dokime diagnose command."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from dokime.quality.diagnose import DiagnoseResult, run_diagnose


def _write_jsonl(docs: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


class TestDiagnoseResult:
    def test_creates_with_required_fields(self):
        result = DiagnoseResult(
            input_path="test.jsonl",
            total_documents=10,
            elapsed_seconds=1.0,
            grade="B+",
            grade_color="green",
            overall_score=82.0,
            quality_distribution={"excellent": 7, "good": 2, "medium": 1, "poor": 0},
            issue_counts={"short_docs": 0, "low_quality": 0},
            worst_documents=[],
            exact_duplicate_count=0,
        )
        assert result.total_documents == 10
        assert result.grade == "B+"
        assert result.minhash_available is False
        assert result.embeddings_available is False

    def test_asdict_serializable(self):
        result = DiagnoseResult(
            input_path="test.jsonl",
            total_documents=5,
            elapsed_seconds=0.5,
            grade="A",
            grade_color="green",
            overall_score=92.0,
            quality_distribution={"excellent": 5, "good": 0, "medium": 0, "poor": 0},
            issue_counts={},
            worst_documents=[(0, 0.9)],
            exact_duplicate_count=0,
        )
        d = asdict(result)
        # Should be JSON-serializable
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["grade"] == "A"


class TestRunDiagnose:
    def test_basic_quality_report(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for text in [
                "This is a normal document with enough text to pass quality checks easily.",
                "Another good document about machine learning and data quality.",
                "A third well-written document covering natural language processing topics.",
            ]:
                f.write(json.dumps({"text": text}) + "\n")
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, quiet=True)

        assert result.total_documents == 3
        assert result.grade in ("A", "B+", "B", "B-", "C+", "C", "D", "F")
        assert 0 <= result.overall_score <= 100
        assert result.elapsed_seconds >= 0
        Path(path).unlink()

    def test_empty_dataset(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, quiet=True)
        assert result.total_documents == 0
        assert result.grade == "N/A"
        Path(path).unlink()

    def test_detects_exact_duplicates(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for text in [
                "This is a unique document about testing.",
                "This is a unique document about testing.",  # exact duplicate
                "A completely different document here.",
            ]:
                f.write(json.dumps({"text": text}) + "\n")
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, quiet=True)
        assert result.exact_duplicate_count == 1
        Path(path).unlink()

    def test_quality_distribution_sums_to_total(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for text in [
                "Good quality document about machine learning and artificial intelligence.",
                "",  # poor
                "Short.",  # likely poor
                "Another well-written document with enough content to be useful for training.",
                "Yet another normal document that should score reasonably well on quality.",
            ]:
                f.write(json.dumps({"text": text}) + "\n")
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, quiet=True)
        dist = result.quality_distribution
        total_from_dist = dist["excellent"] + dist["good"] + dist["medium"] + dist["poor"]
        assert total_from_dist == result.total_documents
        Path(path).unlink()

    def test_worst_documents_ordered(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for text in [
                "A great document about natural language processing with substantial content.",
                "",  # worst
                "###!!!@@@",  # bad
                "Another excellent document covering deep learning and neural networks.",
            ]:
                f.write(json.dumps({"text": text}) + "\n")
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, show_worst=4, quiet=True)
        scores = [s for _, s in result.worst_documents]
        assert scores == sorted(scores)  # ascending order (worst first)
        Path(path).unlink()

    def test_skip_embeddings_flag(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "Test document."}) + "\n")
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, quiet=True)
        assert result.embeddings_available is False
        assert result.outlier_count is None
        assert result.semantic_duplicate_count is None
        Path(path).unlink()

    def test_recommendations_include_dedup_command(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for _ in range(3):
                f.write(json.dumps({"text": "Duplicate document content for testing."}) + "\n")
            f.write(json.dumps({"text": "One unique document."}) + "\n")
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, quiet=True)
        assert result.exact_duplicate_count == 2
        assert any("--dedup" in r for r in result.recommendations)
        Path(path).unlink()

    def test_custom_text_field(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"content": "A document using content field instead of text."}) + "\n")
            f.write(json.dumps({"content": "Another document with a custom field name."}) + "\n")
            path = f.name

        result = run_diagnose(path, text_field="content", skip_embeddings=True, quiet=True)
        assert result.total_documents == 2
        # Should actually score the content, not get 0s
        assert result.overall_score > 0
        Path(path).unlink()

    def test_json_output_is_serializable(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "Test document for JSON output."}) + "\n")
            path = f.name

        result = run_diagnose(path, skip_embeddings=True, quiet=True)
        serialized = json.dumps(asdict(result), default=str)
        parsed = json.loads(serialized)
        assert parsed["total_documents"] == 1
        assert "grade" in parsed
        assert "recommendations" in parsed
        Path(path).unlink()
