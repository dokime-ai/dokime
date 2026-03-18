# Show HN: Dokime -- Open-source data curation toolkit for ML training data

We built Dokime because curating ML training data still means stitching together 5-7 libraries (datasketch for dedup, lingua for language detection, sentence-transformers for embeddings, FAISS for search, datasets for HuggingFace integration) into a one-off pipeline that works on your machine, for your dataset, until it doesn't. Lilac was archived in July 2025. Argilla shifted to annotation. DataTrove is excellent for batch processing but has no interactive exploration. There's a gap for an iterative workbench that helps you figure out what filters to use and what thresholds to set.

Dokime is a single `pip install` that gives you 16 heuristic filters, 3 deduplication methods (exact SHA-256, MinHash-LSH fuzzy, embedding-based semantic), semantic search over your dataset via FAISS, k-NN outlier detection, quality scoring, and HuggingFace Hub integration. It has 10 CLI commands and a Python SDK. Pipelines are configurable via YAML. Execution is streaming -- constant memory regardless of dataset size. Every pipeline run reports per-filter removal counts so you can see exactly what each filter catches and tune accordingly. The test suite has 56 tests. It reads JSONL, Parquet, CSV, and HuggingFace datasets.

The design philosophy is "workbench, not pipeline engine." DataTrove and Data-Juicer are built for scheduled batch runs at scale. Dokime is built for iteration: explore the data, try different thresholds, inspect what gets removed, search for specific content, find outliers, then codify your decisions in a YAML config. If you need Spark-scale processing, use DataTrove. If you need to understand your data and converge on the right curation strategy, that's what Dokime is for. Distributed execution and a web UI are on the roadmap but not shipped yet.

We also ran what we believe is the first independent evaluation of JEPA-SCORE (NeurIPS 2025) for out-of-distribution detection. Short version: k-NN distance beats JEPA-SCORE by 24 AUROC points and is 10,000x faster. Results and code are in the experiments/ directory.

- GitHub: https://github.com/dokime-ai/dokime
- PyPI: https://pypi.org/project/dokime-ai/
- Docs: https://dokime-ai.github.io/dokime
- HuggingFace Space: https://huggingface.co/spaces/dokime-ai/dokime
