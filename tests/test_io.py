"""Tests for readers and writers."""

import json
import tempfile
from pathlib import Path

from dokime.io.readers import read_jsonl
from dokime.io.writers import write_jsonl


class TestJsonlIO:
    def test_roundtrip(self):
        data = [
            {"text": "Hello world", "id": 1},
            {"text": "Another document", "id": 2},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
            path = f.name

        result = list(read_jsonl(path))
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"
        assert result[1]["id"] == 2
        Path(path).unlink()

    def test_write_jsonl(self):
        data = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        count = write_jsonl(data, path)
        assert count == 3

        result = list(read_jsonl(path))
        assert len(result) == 3
        Path(path).unlink()
