<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo.svg" />
    <img src="docs/assets/logo.svg" alt="Dokime logo" width="320" />
  </picture>
</p>

<h3 align="center">The open-source workbench for ML training data.</h3>

<p align="center">
  <a href="https://pypi.org/project/dokime-ai/"><img alt="PyPI" src="https://img.shields.io/pypi/v/dokime-ai?color=blue" /></a>
  <a href="https://pypi.org/project/dokime-ai/"><img alt="Python 3.10+" src="https://img.shields.io/pypi/pyversions/dokime-ai" /></a>
  <a href="https://github.com/dokime-ai/dokime/blob/main/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-green" /></a>
  <a href="https://github.com/dokime-ai/dokime/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/dokime-ai/dokime/ci.yml?label=CI" /></a>
  <a href="https://dokime-ai.github.io/dokime"><img alt="Docs" src="https://img.shields.io/badge/docs-mkdocs-blue" /></a>
  <a href="https://github.com/dokime-ai/dokime/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/dokime-ai/dokime?style=social" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#installation">Install</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="https://dokime-ai.github.io/dokime">Docs</a> &bull;
  <a href="#comparison">Compare</a> &bull;
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

---

Dokime is a Python toolkit for curating ML training data. Filter junk, deduplicate, compute embeddings, find outliers, and search your dataset — from one `pip install`, with a CLI and a Python SDK.

<!-- TODO: Add GIF/screenshot of the web UI once ready -->
<!-- <p align="center"><img src="docs/assets/demo.gif" width="700" /></p> -->

## Installation

```bash
pip install dokime-ai
```

That gives you heuristic filters, exact dedup, and the CLI. For the full toolkit:

```bash
pip install "dokime-ai[all]"    # embeddings, fuzzy dedup, language detection, HF datasets
```

Or pick what you need:

| Extra | What it adds |
|---|---|
| `dokime-ai[embeddings]` | Sentence-transformer embeddings, FAISS search, semantic dedup, anomaly detection |
| `dokime-ai[dedup]` | MinHash-LSH fuzzy deduplication |
| `dokime-ai[nlp]` | Language detection (lingua) |
| `dokime-ai[io]` | HuggingFace `datasets`, Pandas |

## Quick Start

```python
from dokime.core.pipeline import Pipeline
from dokime.core.filters import LengthFilter, WhitespaceFilter, RepetitionFilter
from dokime.quality.dedup import ExactDedup

# Build a pipeline
pipeline = Pipeline("my-curation")
pipeline.add_filter(LengthFilter(min_length=50, max_length=100_000))
pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
pipeline.add_filter(RepetitionFilter(max_repetition_ratio=0.3))
pipeline.add_filter(ExactDedup())

# Run it
result = pipeline.run("data/raw.jsonl", "data/curated.parquet")
print(f"Kept {result['total_kept']:,} / {result['total_read']:,} documents")
```

Or from the command line:

```bash
dokime curate data/raw.jsonl data/clean.parquet \
  --min-length 50 --max-whitespace 0.4 --dedup
```

Or with a YAML config:

```bash
dokime curate data/raw.jsonl data/clean.parquet --config pipeline.yaml
```

See [`examples/quickstart.py`](examples/quickstart.py) for a full walkthrough including embeddings, search, and outlier detection.

## Features

### Heuristic Filters
Twelve built-in filters, all composable, all configurable via code or YAML:

| Filter | What it catches |
|---|---|
| `LengthFilter` | Too short / too long documents |
| `WordCountFilter` | Documents outside a word-count range |
| `LineCountFilter` | Documents outside a line-count range |
| `WhitespaceFilter` | Excessive whitespace (formatting junk) |
| `RepetitionFilter` | Repeated n-grams (boilerplate, spam) |
| `SpecialCharFilter` | Special character overload (encoding artifacts) |
| `AlphaFilter` | Low alphabetic ratio (numeric spam, base64) |
| `URLFilter` | URL-heavy documents (link farms) |
| `StopwordFilter` | Missing stopwords (keyword spam, code) |
| `LanguageFilter` | Wrong language (lingua or fastText backend) |
| `FieldExistsFilter` | Missing required fields |
| `RegexFilter` | Custom pattern matching (include or exclude) |

### Deduplication
- **Exact dedup** (SHA-256) — zero dependencies, streaming
- **Fuzzy dedup** (MinHash-LSH) — catches near-duplicates at scale
- **Semantic dedup** (embedding cosine similarity) — finds paraphrases and reworded copies

### Embeddings & Search
- **Compute embeddings** with any sentence-transformer model
- **Semantic search** — FAISS-backed nearest-neighbor search over your dataset
- **Anomaly/outlier detection** — k-NN distance scoring to surface unusual documents

### Pipeline Orchestration
- Chain any number of filters in a `Pipeline`
- Configure via Python or YAML
- Per-filter removal stats and throughput reporting
- Reads JSONL, Parquet, CSV, and HuggingFace datasets
- Writes JSONL and Parquet

### CLI
Seven commands, zero boilerplate:

```
dokime version     # print version
dokime curate      # run a curation pipeline
dokime stats       # dataset statistics
dokime embed       # compute embeddings
dokime search      # semantic search
dokime outliers    # find anomalous documents
dokime explore     # launch web UI (coming soon)
```

## Comparison

Dokime is a **data workbench** — interactive, exploratory, designed for iteration. Pipeline engines like DataTrove and Data-Juicer are great for scheduled batch processing at massive scale. If you need to understand your data, experiment with filter thresholds, and investigate what you are keeping and discarding, Dokime is the right tool.

| | Dokime | DataTrove | Data-Juicer |
|---|:---:|:---:|:---:|
| Interactive exploration | Yes | No | Limited |
| Semantic search | Yes | No | No |
| Outlier detection | Yes | No | No |
| Embedding-based dedup | Yes | MinHash only | MinHash only |
| CLI + Python SDK | Both | Python only | Both |
| YAML config | Yes | No (Python) | Yes |
| HuggingFace datasets | Yes | Yes | Yes |
| Spark/distributed | Roadmap | Yes | Yes |
| Web UI | Coming soon | No | Yes |

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) — Full workflow: load, filter, dedup, embed, search, outliers
- [`examples/basic_pipeline.yaml`](examples/basic_pipeline.yaml) — Simple YAML config
- [`examples/advanced_pipeline.yaml`](examples/advanced_pipeline.yaml) — All available filters

## Documentation

Full docs at **[dokime-ai.github.io/dokime](https://dokime-ai.github.io/dokime)** (coming soon).

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

## License

Apache 2.0 — see [LICENSE](LICENSE).

<!-- Scarf download tracking pixel — replace PLACEHOLDER with your actual pixel ID from https://scarf.sh -->
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=PLACEHOLDER" />
