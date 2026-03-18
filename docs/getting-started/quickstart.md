# Quick Start

This tutorial walks through the full Dokime workflow: filter, deduplicate, embed, search, and detect outliers.

## 1. Build a pipeline

```python
from dokime.core.pipeline import Pipeline
from dokime.core.filters import LengthFilter, WhitespaceFilter, RepetitionFilter
from dokime.quality.dedup import ExactDedup

pipeline = Pipeline("my-curation")
pipeline.add_filter(LengthFilter(min_length=50, max_length=100_000))
pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
pipeline.add_filter(RepetitionFilter(max_repetition_ratio=0.3))
pipeline.add_filter(ExactDedup())

result = pipeline.run("data/raw.jsonl", "data/curated.jsonl")
print(f"Kept {result['total_kept']} / {result['total_read']} documents")
```

## 2. Same thing from the CLI

```bash
dokime curate data/raw.jsonl data/curated.jsonl \
  --min-length 50 --max-whitespace 0.4 --max-repetition 0.3 --dedup
```

## 3. Or with a YAML config

```yaml
# pipeline.yaml
name: my-curation
filters:
  - LengthFilter:
      min_length: 50
      max_length: 100000
  - WhitespaceFilter:
      max_whitespace_ratio: 0.4
  - RepetitionFilter:
      max_repetition_ratio: 0.3
  - ExactDedup: {}
```

```bash
dokime curate data/raw.jsonl data/curated.jsonl --config pipeline.yaml
```

## 4. Compute embeddings and search

```python
from dokime.embeddings.compute import EmbeddingModel, compute_embeddings
from dokime.embeddings.search import EmbeddingIndex
from dokime.io.readers import auto_read

docs, embeddings = compute_embeddings(
    data=auto_read("data/curated.jsonl"),
    model_name="all-MiniLM-L6-v2",
)

model = EmbeddingModel("all-MiniLM-L6-v2")
index = EmbeddingIndex(embeddings, docs)
results = index.search("how to remove duplicates", model, k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.document['text'][:80]}")
```

## 5. Find outliers

```python
from dokime.embeddings.search import AnomalyScorer

scorer = AnomalyScorer(embeddings)
outlier_indices = scorer.find_outliers(k=10, top_n=5)
```

See the [full examples directory](https://github.com/dokime-ai/dokime/tree/main/examples) for runnable scripts.
