# Reddit Post: r/MachineLearning

**Title:** [P] Dokime: Open-source data curation toolkit for ML training data

**Body:**

We open-sourced Dokime, a Python toolkit for curating ML training data. It consolidates the common data curation stack (heuristic filtering, deduplication, embedding computation, semantic search, outlier detection, quality scoring) into a single library with a CLI and a Python SDK.

**What it does:**

- 16 heuristic filters (length, word count, whitespace ratio, repeated n-grams, special characters, alphabetic ratio, URL density, stopword presence, language detection, regex patterns, field existence)
- 3 deduplication methods: exact (SHA-256, streaming), fuzzy (MinHash-LSH), and semantic (embedding cosine similarity)
- Embedding computation via sentence-transformers, FAISS-backed semantic search, k-NN anomaly/outlier detection
- Quality scoring that adds per-document signals (entropy, alpha ratio, estimated token count, composite quality score)
- YAML-configured pipelines with per-filter removal statistics
- Reads JSONL, Parquet, CSV, HuggingFace datasets. Writes JSONL and Parquet. Push to HuggingFace Hub via CLI.
- 10 CLI commands: `version`, `curate`, `stats`, `embed`, `search`, `outliers`, `score`, `push`, `attribute`, `explore`
- 56 tests

**Installation:**

```bash
pip install dokime            # core: filters, exact dedup, CLI
pip install "dokime[all]"     # everything: embeddings, fuzzy dedup, language detection, HF datasets
```

**Quick example:**

```bash
dokime curate data/raw.jsonl data/clean.parquet \
  --min-length 50 --max-whitespace 0.4 --dedup

dokime embed data/clean.parquet embeddings.npy --model all-MiniLM-L6-v2

dokime search data/clean.parquet "examples of data augmentation" \
  --embeddings embeddings.npy

dokime outliers data/clean.parquet --embeddings embeddings.npy --top 20
```

**Why we built it:**

Data curation typically involves stitching together datasketch, lingua, sentence-transformers, FAISS, and the HuggingFace datasets library into a bespoke pipeline. Each library has its own data format expectations and configuration approach. The pipeline is hard to reproduce, hard to hand off, and the rationale behind filter thresholds is rarely recorded.

The existing tool landscape has gaps. Lilac (interactive data tool) was archived in July 2025. Argilla has shifted focus to annotation/RLHF. DataTrove is excellent for large-scale batch processing but has no semantic search, no outlier detection, and is Python-only with no YAML config. Data-Juicer has extensive filters but high setup overhead. NeMo Curator is tightly coupled to NVIDIA GPUs.

Dokime is designed as a **data workbench** rather than a pipeline engine. The difference: a pipeline engine is optimized for scheduled batch runs at scale; a workbench is optimized for iteration. You explore the data, try different filter thresholds, inspect what gets removed, search for specific content, find outliers, and gradually converge on a curation strategy. Then you lock it down in a YAML config and run it in batch. Per-filter removal statistics are printed on every run so you can see exactly what each filter catches.

**Honest comparison:**

| | Dokime | DataTrove | Data-Juicer |
|---|---|---|---|
| Heuristic filters | 16 | ~20 | ~50 |
| Semantic search | Yes | No | No |
| Outlier detection | Yes | No | No |
| Embedding-based dedup | Yes | MinHash only | MinHash only |
| Quality scoring | Yes | No | Limited |
| CLI + Python SDK | Both | Python only | Both |
| YAML configuration | Yes | No | Yes |
| Distributed/Spark | Not yet | Yes | Yes |
| Web UI | Roadmap | No | Yes |

DataTrove and Data-Juicer have more filters and distributed execution. If you need petabyte-scale processing, those are better choices. Dokime targets datasets that fit on one machine (up to tens of millions of documents) where a human is iterating on quality thresholds.

**JEPA-SCORE benchmark (bonus):**

We ran what we believe is the first independent evaluation of JEPA-SCORE (arXiv:2510.05949, NeurIPS 2025) for OOD detection. We compared it against k-NN distance, Mahalanobis distance, and Isolation Forest on CIFAR-10 vs SVHN using DINOv2 ViT-S/14.

Results: k-NN achieves 0.9652 AUROC in <1 second. JEPA-SCORE achieves 0.7285 AUROC in 32 minutes. Mahalanobis hits 1.0000 AUROC. The original paper contained no baseline comparisons. Caveats: 500 samples per split, one encoder, well-separated OOD pair. Full results and code in the repo under `experiments/jepa_score/`.

**What's missing:**

- No distributed execution (Spark/Ray) -- on the roadmap
- Web UI (`dokime explore`) is a placeholder -- actively building it
- 16 filters is fewer than DataTrove (~20) and Data-Juicer (~50)
- Data attribution (`dokime attribute` via TRAK) works but is computationally expensive and limited to ~50K examples on a single GPU

Contributions welcome. Apache 2.0 license.

**Links:**

- GitHub: https://github.com/dokime-ai/dokime
- PyPI: https://pypi.org/project/dokime/
- Docs: https://dokime-ai.github.io/dokime
- HuggingFace Space: https://huggingface.co/spaces/dokime-ai/dokime
