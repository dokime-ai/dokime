# Data Attribution

Dokime's attribution engine answers: **"Which training examples help or hurt your model?"**

It uses [TRAK](https://github.com/MadryLab/trak) to compute per-example influence scores. Requires `dokime[attribution]` (installs torch, transformers, traker).

## Python API

```python
from dokime.attribution.engine import AttributionEngine

engine = AttributionEngine(
    model_name="gpt2",
    train_data="train.jsonl",
    eval_data="eval.jsonl",
)

scores = engine.compute()
# scores.shape = (n_eval, n_train)
# Positive = training example helps eval performance
# Negative = training example hurts eval performance

harmful = engine.find_harmful(top_n=50)   # [(train_idx, score), ...]
helpful = engine.find_helpful(top_n=50)
engine.print_summary()
```

## CLI

```bash
dokime attribute train.jsonl eval.jsonl \
  --model gpt2 \
  --top 20 \
  --save-dir ./attribution_results
```

## How it works

1. A proxy model (default: GPT-2) is loaded
2. Training examples are featurized using TRAK's random projection
3. Each eval example is scored against all training examples
4. Positive scores mean the training example improves performance on that eval example; negative scores mean it hurts

## When to use this

- **Cleaning fine-tuning data**: remove harmful examples before training
- **Debugging model failures**: trace bad outputs back to training data
- **Data valuation**: understand which examples matter most

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `gpt2` | HuggingFace model for proxy |
| `max_length` | `512` | Max token length |
| `proj_dim` | `2048` | TRAK projection dimension |
| `device` | auto | `cpu` or `cuda` |
| `save_dir` | temp dir | Where to save TRAK artifacts |
