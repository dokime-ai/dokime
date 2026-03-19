# CLI Reference

All commands are available via the `dokime` command after installation.

## dokime version

Print the installed version.

```bash
dokime version
```

## dokime curate

Run a curation pipeline on a dataset.

```bash
# With flags
dokime curate input.jsonl output.parquet \
  --min-length 50 --max-length 100000 \
  --max-whitespace 0.4 --max-repetition 0.3 --max-special 0.3 \
  --dedup --fuzzy-dedup 0.8

# With YAML config
dokime curate input.jsonl output.parquet --config pipeline.yaml
```

| Flag | Description |
|---|---|
| `--config`, `-c` | Path to YAML pipeline config |
| `--min-length` | Minimum document length (chars) |
| `--max-length` | Maximum document length (chars) |
| `--max-whitespace` | Max whitespace ratio (0-1) |
| `--max-repetition` | Max n-gram repetition ratio (0-1) |
| `--max-special` | Max special character ratio (0-1) |
| `--dedup` | Enable exact deduplication |
| `--fuzzy-dedup` | MinHash dedup threshold (0-1) |

## dokime stats

Show basic statistics about a dataset.

```bash
dokime stats data/raw.jsonl
```

Outputs document count, total characters, average/min/max length.

## dokime embed

Compute embeddings and save as `.npy`. Requires `dokime[embeddings]`.

```bash
dokime embed data/curated.jsonl embeddings.npy \
  --model all-MiniLM-L6-v2 --batch-size 64 --device cuda
```

## dokime search

Semantic search over a dataset. Requires `dokime[embeddings]`.

```bash
dokime search data/curated.jsonl "your query here" -k 10

# With precomputed embeddings (faster)
dokime search data/curated.jsonl "your query" -e embeddings.npy
```

## dokime outliers

Find anomalous documents via k-NN distance. Requires `dokime[embeddings]`.

```bash
dokime outliers data/curated.jsonl --top 20 -k 10

# With precomputed embeddings
dokime outliers data/curated.jsonl -e embeddings.npy --top 20
```

## dokime attribute

Score training data influence on model performance. Requires `dokime[attribution]`.

```bash
dokime attribute train.jsonl eval.jsonl \
  --model gpt2 --top 20 --save-dir ./results
```

## dokime explore

Launch an interactive web UI. Requires `dokime[explore]`.

```bash
dokime explore data/curated.jsonl --port 8765 -e embeddings.npy
```
