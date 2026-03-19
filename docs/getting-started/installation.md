# Installation

## Base install

```bash
pip install dokime
```

This gives you heuristic filters, exact deduplication, the CLI, and JSONL/Parquet/CSV I/O. No heavy dependencies.

## Extras

Install only what you need:

```bash
pip install "dokime[embeddings]"   # sentence-transformers, FAISS search, semantic dedup
pip install "dokime[dedup]"        # MinHash-LSH fuzzy deduplication
pip install "dokime[nlp]"          # language detection (lingua)
pip install "dokime[io]"           # HuggingFace datasets, Pandas
pip install "dokime[attribution]"  # TRAK-based data attribution (torch, transformers)
pip install "dokime[explore]"      # web UI (FastAPI)
```

Or everything at once:

```bash
pip install "dokime[all]"
```

## Development

```bash
git clone https://github.com/dokime-ai/dokime.git
cd dokime
pip install -e ".[dev]"
pre-commit install
```

## Requirements

- Python 3.10+
- No GPU required (but recommended for embeddings and attribution)
