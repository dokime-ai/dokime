# Embeddings

Dokime uses sentence-transformers to compute embeddings and FAISS for fast similarity search. Requires `dokime[embeddings]`.

## Compute embeddings

```python
from dokime.embeddings.compute import compute_embeddings
from dokime.io.readers import auto_read

docs, embeddings = compute_embeddings(
    data=auto_read("data/curated.jsonl"),
    model_name="all-MiniLM-L6-v2",
    batch_size=64,
)
# embeddings.shape = (n_docs, 384)
```

Save to disk:

```bash
dokime embed data/curated.jsonl embeddings.npy --model all-MiniLM-L6-v2
```

## Semantic search

Find documents similar to a natural language query:

```python
from dokime.embeddings.compute import EmbeddingModel
from dokime.embeddings.search import EmbeddingIndex

model = EmbeddingModel("all-MiniLM-L6-v2")
index = EmbeddingIndex(embeddings, docs)
results = index.search("data deduplication methods", model, k=5)

for r in results:
    print(f"[{r.score:.3f}] {r.document['text'][:80]}")
```

Or from the CLI:

```bash
dokime search data/curated.jsonl "data deduplication methods" -k 5
```

## Outlier detection

Surface anomalous documents using k-NN distance scoring:

```python
from dokime.embeddings.search import AnomalyScorer

scorer = AnomalyScorer(embeddings)
outlier_indices = scorer.find_outliers(k=10, top_n=20)
scores = scorer.score_all(k=10)
```

```bash
dokime outliers data/curated.jsonl --top 20
```

## Semantic deduplication

Find near-duplicate documents by embedding cosine similarity:

```python
from dokime.embeddings.dedup import find_semantic_duplicates

pairs = find_semantic_duplicates(embeddings, docs, threshold=0.90)
for idx_a, idx_b, similarity in pairs:
    print(f"Pair ({idx_a}, {idx_b}): {similarity:.3f}")
```
