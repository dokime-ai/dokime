# Pipelines

Pipelines chain filters together and run them over a dataset. You can build them in Python or load from YAML.

## Python API

```python
from dokime.core.pipeline import Pipeline
from dokime.core.filters import LengthFilter, WhitespaceFilter
from dokime.quality.dedup import ExactDedup

pipeline = Pipeline("my-pipeline")
pipeline.add_filter(LengthFilter(min_length=50))
pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
pipeline.add_filter(ExactDedup())

result = pipeline.run("input.jsonl", "output.parquet")
```

`run()` returns a stats dict with `total_read`, `total_kept`, `total_removed`, `removal_rate_pct`, `per_filter_removed`, and `docs_per_second`.

### Dry run

Pass `output_path=None` to run without writing:

```python
stats = pipeline.run("input.jsonl", None)
```

## YAML config

```yaml
name: my-pipeline
filters:
  - LengthFilter:
      min_length: 50
      max_length: 100000
  - WhitespaceFilter:
      max_whitespace_ratio: 0.4
  - ExactDedup: {}
```

Load and run:

```python
pipeline = Pipeline.from_config("pipeline.yaml")
pipeline.run("input.jsonl", "output.parquet")
```

Or from the CLI:

```bash
dokime curate input.jsonl output.parquet --config pipeline.yaml
```

## Supported formats

| Format | Read | Write |
|---|---|---|
| JSONL | Yes | Yes |
| Parquet | Yes | Yes |
| CSV | Yes | -- |
| HuggingFace datasets | Yes (with `[io]`) | -- |

Output format is inferred from the file extension (`.parquet` or `.jsonl`).

## Custom filters

Any class inheriting from `Filter` works in a pipeline. Register it for YAML support:

```python
from dokime.core.filters import Filter
from dokime.core.registry import register_filter

class MyFilter(Filter):
    def filter(self, sample):
        return len(sample.get("text", "")) > 0
    def name(self):
        return "MyFilter"

register_filter("MyFilter", MyFilter)
```
