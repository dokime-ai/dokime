# Twitter Threads

---

## Thread 1: Launch Announcement

**Tweet 1:**
We just open-sourced Dokime -- a Python toolkit for curating ML training data.

16 filters. 3 dedup methods. Semantic search. Outlier detection. Quality scoring. One pip install. 10 CLI commands.

GitHub: https://github.com/dokime-ai/dokime

**Tweet 2:**
The problem: curating training data means stitching together datasketch + lingua + sentence-transformers + FAISS + datasets into a one-off script that only you understand.

Dokime puts all of that behind a single library with a CLI and a Python SDK.

**Tweet 3:**
Quick example -- filter and deduplicate a JSONL dataset from the command line:

```
dokime curate data/raw.jsonl data/clean.parquet \
  --min-length 50 --max-whitespace 0.4 --dedup
```

Every run prints per-filter removal counts so you can see exactly what each filter is catching.

**Tweet 4:**
Then explore your data with embeddings:

```
dokime embed data/clean.parquet embeddings.npy
dokime search data/clean.parquet "data augmentation" --embeddings embeddings.npy
dokime outliers data/clean.parquet --embeddings embeddings.npy --top 20
```

Semantic search and k-NN outlier detection, backed by FAISS.

**Tweet 5:**
Dokime is a workbench, not a pipeline engine. DataTrove and Data-Juicer are great for batch processing at scale. Dokime is for iteration: explore, adjust thresholds, inspect what gets removed, converge on the right strategy, then codify it in YAML.

**Tweet 6:**
What's missing (we'll be honest):
- No distributed/Spark support yet
- Web UI is on the roadmap but not shipped
- 16 filters vs Data-Juicer's 50

We're early. Contributions welcome. Apache 2.0.

**Tweet 7:**
Links:
- GitHub: https://github.com/dokime-ai/dokime
- PyPI: https://pypi.org/project/dokime-ai/
- Docs: https://dokime-ai.github.io/dokime
- HuggingFace Space: https://huggingface.co/spaces/dokime-ai/dokime

```
pip install dokime-ai
```

---

## Thread 2: The Problem

**Tweet 1:**
Data curation tooling for ML is in a strange place.

We have mature tools for experiment tracking (W&B, MLflow), training (PyTorch, JAX), serving (vLLM, TGI), and evaluation (lm-eval-harness, HELM).

But for data curation -- the step that arguably matters most -- it's still ad-hoc scripts and Jupyter notebooks.

**Tweet 2:**
A typical data curation workflow:

1. Write a script to filter by length
2. Add datasketch for MinHash dedup
3. Add lingua for language detection
4. Add sentence-transformers + FAISS for semantic exploration
5. Add datasets for HuggingFace Hub

Five libraries, five config styles, zero reproducibility.

**Tweet 3:**
The landscape is not helping.

Lilac (interactive data tool) -- archived July 2025.
Argilla -- shifted focus to annotation/RLHF.
DataTrove -- great batch processing, no interactive exploration.
Data-Juicer -- kitchen-sink approach, high setup cost.
NeMo Curator -- tightly coupled to NVIDIA GPUs.

**Tweet 4:**
The gap is not for another framework that runs filters at scale. DataTrove does that well.

The gap is for a workbench that helps you figure out *what* to filter, *what* thresholds to use, and *what* your data looks like before and after. Interactive, iterative, reproducible.

**Tweet 5:**
That's why we built Dokime. Open-source, Apache 2.0.

One install. 16 filters. Semantic search. Outlier detection. Per-filter removal stats. YAML configs. HuggingFace integration.

https://github.com/dokime-ai/dokime

---

## Thread 3: JEPA-SCORE Benchmark

**Tweet 1:**
We tested JEPA-SCORE (arXiv:2510.05949, NeurIPS 2025) for out-of-distribution detection and compared it against simple baselines.

Results: k-NN distance beats JEPA-SCORE by 24 AUROC points and is 10,000x faster.

This is, as far as we know, the first independent evaluation.

**Tweet 2:**
Setup:
- In-distribution: CIFAR-10 (500 samples)
- Out-of-distribution: SVHN (500 samples)
- Encoder: DINOv2 ViT-S/14

Methods tested:
- JEPA-SCORE (randomized SVD of encoder Jacobian)
- k-NN distance
- Mahalanobis distance
- Isolation Forest

**Tweet 3:**
Results:

| Method | AUROC | Time |
|---|---|---|
| Mahalanobis | 1.0000 | 0.2s |
| k-NN (k=10) | 0.9652 | 0.0s |
| JEPA-SCORE | 0.7285 | 1926s |
| Isolation Forest | 0.4387 | 0.2s |

JEPA-SCORE requires 64 backward passes per sample to compute the encoder Jacobian. k-NN just measures embedding distance.

**Tweet 4:**
Caveats (important):
- 500 samples per split -- a larger study is needed
- Only one encoder tested (DINOv2 ViT-S/14)
- CIFAR-10 vs SVHN is a well-separated OOD pair
- We used randomized SVD, not the full Jacobian

The original paper had no baseline comparisons. We think these are worth reporting even at this scale.

**Tweet 5:**
What this means for us: Dokime's outlier detection uses k-NN distance in embedding space. This experiment confirmed it was the right call.

Full experiment code and results: https://github.com/dokime-ai/dokime/tree/main/experiments/jepa_score
