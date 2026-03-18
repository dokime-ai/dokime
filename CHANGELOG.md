# Changelog

All notable changes to Dokime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Embedding computation** — `dokime embed` CLI command and `EmbeddingModel` / `compute_embeddings()` API for sentence-transformer embeddings (any model from HuggingFace). Saves embeddings as `.npy` files.
- **Semantic search** — `dokime search` CLI command and `EmbeddingIndex` class for FAISS-backed nearest-neighbor search over datasets. Query with natural language, get ranked results.
- **Outlier/anomaly detection** — `dokime outliers` CLI command and `AnomalyScorer` class. Uses k-NN distance in embedding space to surface unusual documents.
- **Semantic deduplication** — `find_semantic_duplicates()` and `deduplicate_by_embeddings()` in `dokime.embeddings.dedup`. Finds near-duplicates that hash-based methods miss (paraphrases, reworded copies).
- **6 new heuristic filters**: `WordCountFilter`, `LineCountFilter`, `AlphaFilter`, `URLFilter`, `StopwordFilter`, `FieldExistsFilter`
- **Language detection filter** — `LanguageFilter` with dual-backend support (lingua-language-detector primary, fastText fallback). ISO 639-1 language codes, configurable confidence threshold.
- **Regex filter** — `RegexFilter` for custom pattern-based include/exclude filtering
- **`dokime stats` command** — show dataset statistics (document count, character counts, min/max/avg length) with rich table output
- **Filter registry** — `FILTER_REGISTRY` and `register_filter()` for YAML config support and custom filter registration
- **YAML pipeline configuration** — `Pipeline.from_config()` loads filter chains from YAML files
- **Per-filter removal stats** — pipeline execution reports how many documents each filter removed
- **Examples** — `examples/quickstart.py` (full workflow), `examples/basic_pipeline.yaml`, `examples/advanced_pipeline.yaml` (all filters)

### Initial (project scaffold)
- CLI with `dokime curate` and `dokime explore` commands
- Heuristic filters: `LengthFilter`, `WhitespaceFilter`, `RepetitionFilter`, `SpecialCharFilter`
- Exact deduplication (SHA-256) — `ExactDedup`
- MinHash-LSH fuzzy deduplication — `MinHashDedup`
- Readers: JSONL, Parquet, CSV, HuggingFace datasets (streaming)
- Writers: JSONL, Parquet
- Pipeline abstraction for composable curation workflows
- Optional dependency extras: `[io]`, `[nlp]`, `[embeddings]`, `[dedup]`, `[all]`, `[dev]`, `[docs]`
