# Dokime

**pytest for your training data.**

One command to grade your ML training data. Filter junk, deduplicate, score quality, find outliers, and search -- from a single `pip install`.

```bash
pip install dokime
```

## Score your data in 10 seconds

```bash
dokime score data.jsonl
```

## Build a curation pipeline

```python
from dokime.core.pipeline import Pipeline
from dokime.core.filters import LengthFilter, WhitespaceFilter, RepetitionFilter
from dokime.quality.dedup import ExactDedup

pipeline = Pipeline("my-curation")
pipeline.add_filter(LengthFilter(min_length=50, max_length=100_000))
pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
pipeline.add_filter(RepetitionFilter(max_repetition_ratio=0.3))
pipeline.add_filter(ExactDedup())

result = pipeline.run("data/raw.jsonl", "data/curated.parquet")
```

Or from the CLI:

```bash
dokime curate data/raw.jsonl data/clean.parquet --min-length 50 --dedup
```

## Next steps

- [Installation](getting-started/installation.md) -- all extras explained
- [Quick Start](getting-started/quickstart.md) -- end-to-end tutorial
- [Filters](user-guide/filters.md) -- all 12 built-in filters
- [Pipelines](user-guide/pipelines.md) -- Python API and YAML config
- [Embeddings](user-guide/embeddings.md) -- search, outliers, semantic dedup
- [Attribution](user-guide/attribution.md) -- find which training examples help or hurt
- [CLI Reference](user-guide/cli.md) -- all 10 commands
