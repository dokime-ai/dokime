# Dokime

**The open-source workbench for ML training data.**

Dokime lets you filter, deduplicate, embed, search, and score your training data -- from one `pip install`.

```bash
pip install dokime-ai
```

## What can it do?

```python
from dokime.core.pipeline import Pipeline
from dokime.core.filters import LengthFilter, WhitespaceFilter
from dokime.quality.dedup import ExactDedup

pipeline = Pipeline("my-curation")
pipeline.add_filter(LengthFilter(min_length=50))
pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
pipeline.add_filter(ExactDedup())

result = pipeline.run("data/raw.jsonl", "data/clean.parquet")
```

Or from the CLI:

```bash
dokime curate data/raw.jsonl data/clean.parquet --min-length 50 --dedup
```

## Next steps

- [Installation](getting-started/installation.md) -- all extras explained
- [Quick Start](getting-started/quickstart.md) -- end-to-end tutorial
- [Filters](user-guide/filters.md) -- all 12+ built-in filters
- [Pipelines](user-guide/pipelines.md) -- Python API and YAML config
- [Embeddings](user-guide/embeddings.md) -- search, outliers, semantic dedup
- [Attribution](user-guide/attribution.md) -- find which training examples help or hurt
- [CLI Reference](user-guide/cli.md) -- all 8 commands
