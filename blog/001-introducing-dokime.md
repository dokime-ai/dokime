# Introducing Dokime: The Open-Source Workbench for ML Training Data

*March 2026*

Everyone agrees that data quality determines model quality. So why is the state of the art for curating training data still a spreadsheet and a prayer?

---

## The Problem No One Wants to Talk About

A CHI 2025 study found that ML practitioners doing data curation overwhelmingly rely on ad-hoc tooling: spreadsheets, one-off scripts, Jupyter notebooks with 300 cells, and manual inspection of random samples. The researchers called it "artisanal data work." That is a polite way of saying we are flying blind.

This is strange, if you think about it. We have mature, well-funded tooling for every other part of the ML lifecycle. Experiment tracking has Weights & Biases and MLflow. Model training has PyTorch and JAX. Serving has vLLM and TGI. Evaluation has lm-eval-harness and HELM. But data curation -- the step that arguably determines model quality more than any other -- is still a folder of scripts that one person wrote and no one else understands.

Here is what a typical data curation workflow looks like in practice. You start by writing a Python script to filter by length. Then you realize you need deduplication, so you pull in `datasketch` for MinHash. Then someone asks "what languages are in this dataset?" so you add `lingua` or `fastText`. Then you want to explore the data semantically, so you compute embeddings with `sentence-transformers` and build a FAISS index. Then you want to push the result to HuggingFace Hub, so you add `datasets`. Then someone new joins the team and asks "why did we filter at 50 characters and not 100?" and nobody knows, because the rationale was never recorded and the experiment was never reproduced.

By the end, you have cobbled together 5-7 separate libraries, each with its own data format expectations, its own configuration style, and its own assumptions about how your data is structured. The pipeline is a ball of duct tape that works on your machine, for this dataset, until it does not.

The tooling landscape is not helping. Lilac, which was the most promising interactive data tool, was archived in July 2025. Argilla has shifted focus to annotation and RLHF -- it is in maintenance mode for general data curation. DataTrove is excellent for massive batch processing, but it has no UI, no semantic search, and is Python-only with no YAML configuration. Data-Juicer has the kitchen-sink approach but requires significant setup and its documentation assumes you already know what you want. NeMo Curator is powerful but GPU-heavy and tightly coupled to the NVIDIA ecosystem.

There is a gap. Not for another pipeline framework that runs filters at scale -- DataTrove does that well. The gap is for an interactive workbench that helps you figure out *what* filters to run, *what* thresholds to set, and *what* your data actually looks like before and after curation. A tool that makes data curation decisions visible, reproducible, and iterative.

That is what we built.

## What Dokime Does

Dokime is a Python toolkit for curating ML training data. One install. Ten CLI commands. Sixteen filters. Three dedup methods. Embedding-based semantic search and outlier detection. Quality scoring. HuggingFace Hub integration. YAML configuration. And a Python SDK for when you need to go beyond the CLI.

```bash
pip install dokime
```

That gives you heuristic filters, exact dedup, and the CLI. For the full toolkit:

```bash
pip install "dokime[all]"
```

The quickest way to see what Dokime does is to run it:

```bash
# Filter and deduplicate a JSONL dataset
dokime curate data/raw.jsonl data/clean.parquet \
  --min-length 50 --max-whitespace 0.4 --dedup

# See what you have
dokime stats data/clean.parquet

# Compute embeddings
dokime embed data/clean.parquet embeddings.npy --model all-MiniLM-L6-v2

# Search your data with natural language
dokime search data/clean.parquet "examples of data augmentation" \
  --embeddings embeddings.npy

# Find the weirdest documents
dokime outliers data/clean.parquet --embeddings embeddings.npy --top 20

# Add quality scores to every document
dokime score data/clean.parquet data/scored.parquet

# Push to HuggingFace Hub
dokime push data/clean.parquet your-username/my-curated-dataset
```

Or from Python:

```python
from dokime.core.pipeline import Pipeline
from dokime.core.filters import LengthFilter, WhitespaceFilter, RepetitionFilter
from dokime.quality.dedup import ExactDedup

pipeline = Pipeline("my-curation")
pipeline.add_filter(LengthFilter(min_length=50, max_length=100_000))
pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
pipeline.add_filter(RepetitionFilter(max_repetition_ratio=0.3))
pipeline.add_filter(ExactDedup())

result = pipeline.run("data/raw.jsonl", "data/curated.parquet")
print(f"Kept {result['total_kept']:,} / {result['total_read']:,} documents")
```

The pipeline streams documents one at a time through the filter chain. It never loads the full dataset into memory. When it finishes, it prints a summary showing exactly how many documents each filter removed, so you know where the bodies are buried.

## How It Works: A Real Example

Let us walk through curating a messy JSONL dataset from scratch. Suppose you have scraped 100,000 web documents into `raw.jsonl` and you want to produce a clean training set.

### Step 1: Define your pipeline in YAML

```yaml
# pipeline.yaml
name: web-curation-v1

filters:
  # Require that a "text" field exists and is non-empty
  - FieldExistsFilter:
      required_field: text

  # Drop documents shorter than 50 chars or longer than 100K chars
  - LengthFilter:
      min_length: 50
      max_length: 100000

  # Drop documents with fewer than 10 words
  - WordCountFilter:
      min_words: 10
      max_words: 50000

  # Excessive whitespace = formatting junk
  - WhitespaceFilter:
      max_whitespace_ratio: 0.4

  # Repeated 5-grams = boilerplate, spam, template text
  - RepetitionFilter:
      max_repetition_ratio: 0.3
      ngram_size: 5

  # Too many special characters = encoding artifacts, base64
  - SpecialCharFilter:
      max_special_ratio: 0.3

  # Too few alphabetic characters = numeric spam, code dumps
  - AlphaFilter:
      min_alpha_ratio: 0.5

  # URL-heavy documents = link farms, navigation pages
  - URLFilter:
      max_url_ratio: 0.1

  # Missing stopwords = keyword stuffing, SEO spam
  - StopwordFilter:
      min_stopword_ratio: 0.05

  # Remove cookie banners, ToS boilerplate
  - RegexFilter:
      pattern: "(?i)(cookie policy|privacy notice|terms of service)"
      exclude: true

  # English only
  - LanguageFilter:
      languages: [en]
      min_confidence: 0.5

  # Exact dedup (SHA-256, zero dependencies, streaming)
  - ExactDedup: {}

  # Fuzzy dedup (MinHash-LSH, catches near-duplicates)
  - MinHashDedup:
      threshold: 0.8
      num_perm: 128
```

### Step 2: Run it

```bash
dokime curate raw.jsonl curated.parquet --config pipeline.yaml
```

Dokime streams through the input, applying each filter in order. Each document is tested against the chain and either kept or discarded. The first filter that rejects a document gets credited with the removal, so you can see exactly what is catching what.

The output looks like this:

```
Dokime -- Running pipeline web-curation-v1
  Input:   raw.jsonl
  Output:  curated.parquet
  Filters: 14

Processing: 100000 docs [00:47, 2127.66 docs/s]

                    Pipeline Results
  +-----------------------+------------------+
  | Metric                |            Value |
  +-----------------------+------------------+
  | Documents read        |          100,000 |
  | Documents kept        |           61,284 |
  | Documents removed     |           38,716 |
  | Removal rate          |           38.72% |
  | Elapsed               |            47.0s |
  | Throughput            |    2,128 docs/s  |
  +-----------------------+------------------+

                  Per-Filter Removals
  +-----------------------------------+---------+
  | Filter                            | Removed |
  +-----------------------------------+---------+
  | FieldExistsFilter(field=text)     |     312 |
  | LengthFilter(min=50, max=100000)  |   4,891 |
  | WordCountFilter(min=10, max=50K)  |   2,104 |
  | WhitespaceFilter(max_ratio=0.4)   |   1,567 |
  | RepetitionFilter(max_ratio=0.3)   |   6,203 |
  | SpecialCharFilter(max_ratio=0.3)  |     892 |
  | AlphaFilter(min_ratio=0.5)        |   1,445 |
  | URLFilter(max_ratio=0.1)          |   2,871 |
  | StopwordFilter(min_ratio=0.05)    |     704 |
  | RegexFilter(exclude)              |   1,388 |
  | LanguageFilter(langs=[en])        |   5,612 |
  | ExactDedup                        |   8,104 |
  | MinHashDedup(threshold=0.8)       |   2,623 |
  +-----------------------------------+---------+
```

This per-filter breakdown is the most important part. Without it, you have no idea whether your length filter is too aggressive, whether dedup is catching enough, or whether the language filter is rejecting documents it should keep. Every data curation project involves tuning these thresholds, and you cannot tune what you cannot measure.

### Step 3: Explore with embeddings

Now you have 61,284 clean documents. But are they diverse? Are there clusters of near-identical content that fuzzy dedup missed? Are there outliers that should not be there?

```python
from dokime.embeddings.compute import compute_embeddings
from dokime.embeddings.search import EmbeddingIndex, AnomalyScorer
from dokime.embeddings.compute import EmbeddingModel
from dokime.io.readers import auto_read

# Compute embeddings (uses sentence-transformers under the hood)
documents, embeddings = compute_embeddings(
    data=auto_read("curated.parquet"),
    model_name="all-MiniLM-L6-v2",
    batch_size=64,
)
# embeddings.shape: (61284, 384)

# Semantic search: find documents about a specific topic
model = EmbeddingModel("all-MiniLM-L6-v2")
index = EmbeddingIndex(embeddings, documents)
results = index.search("data augmentation techniques for NLP", model, k=10)

for i, r in enumerate(results, 1):
    print(f"{i}. [{r.score:.3f}] {r.document['text'][:100]}...")
```

The search is backed by FAISS, so it is fast even on hundreds of thousands of documents. You get ranked results by cosine similarity, which lets you quickly verify that your dataset actually contains what you think it contains.

### Step 4: Find outliers

```python
scorer = AnomalyScorer(embeddings)
scores = scorer.score_all(k=10)

# Higher score = more unusual (farther from neighbors in embedding space)
outlier_indices = scorer.find_outliers(k=10, top_n=50)
for idx in outlier_indices:
    print(f"[{scores[idx]:.3f}] {documents[idx]['text'][:100]}...")
```

Outlier detection uses k-nearest-neighbor distance in embedding space. Documents that are far from their neighbors are likely mislabeled, corrupted, or just weird. In our experience, the top outliers almost always reveal something interesting -- a document in the wrong language that slipped past the language filter, a boilerplate page with enough real text to pass heuristics, or a genuinely unusual document that you want to inspect manually.

### Step 5: Semantic dedup (catch what MinHash missed)

```python
from dokime.embeddings.dedup import find_semantic_duplicates

pairs = find_semantic_duplicates(embeddings, documents, threshold=0.92)
print(f"Found {len(pairs)} near-duplicate pairs by meaning")

for idx_a, idx_b, similarity in pairs[:5]:
    print(f"\nPair (similarity={similarity:.3f}):")
    print(f"  A: {documents[idx_a]['text'][:80]}...")
    print(f"  B: {documents[idx_b]['text'][:80]}...")
```

MinHash dedup catches documents that share surface-level n-grams. Semantic dedup catches documents that say the same thing in different words -- paraphrases, reworded copies, translated-and-back content. Hash-based methods are blind to these. Embedding-based similarity is not.

### Step 6: Quality scoring

```bash
dokime score curated.parquet scored.parquet
```

This adds quality signals to every document without removing anything: character entropy, alphabetic ratio, whitespace ratio, estimated token count, average word length, and a composite quality score (0-1). The scored output looks like this:

```json
{
  "text": "Machine learning models require large amounts of ...",
  "_char_count": 247,
  "_word_count": 38,
  "_estimated_tokens": 62,
  "_char_entropy": 4.312,
  "_whitespace_ratio": 0.153,
  "_alpha_ratio": 0.821,
  "_special_ratio": 0.024,
  "_avg_word_length": 5.3,
  "_quality_score": 0.875
}
```

You can then sort by `_quality_score`, visualize distributions, set thresholds for downstream filtering, or use the scores as sample weights during training. The composite score combines entropy, alphabetic ratio, word count, and average word length into a single number. It is a blunt instrument -- you will probably want to define your own scoring logic for your domain -- but it is a useful starting point for quickly identifying the best and worst documents in a corpus.

## What Makes Dokime Different

There are good tools in this space. We are not pretending otherwise. Here is an honest comparison:

| | **Dokime** | **DataTrove** | **Data-Juicer** |
|---|:---:|:---:|:---:|
| One `pip install` | Yes | Yes | Partial |
| Heuristic filters | 16 | ~20 | ~50 |
| Semantic search | Yes | No | No |
| Outlier detection | Yes | No | No |
| Embedding-based dedup | Yes | MinHash only | MinHash only |
| Quality scoring | Yes | No | Limited |
| CLI + Python SDK | Both | Python only | Both |
| YAML configuration | Yes | No (Python) | Yes |
| HuggingFace Hub push | Yes | Yes | Yes |
| Data attribution | Yes (TRAK) | No | No |
| Spark/distributed | Not yet | Yes | Yes |
| Web UI | Coming soon | No | Yes |

DataTrove has more heuristic filters than we do, and it has Spark support for distributed processing. Data-Juicer has even more filters and a web UI. If you are processing petabytes on a cluster, those tools are the right choice.

Dokime is a **data workbench**, not a pipeline engine. The difference matters. A pipeline engine is optimized for scheduled batch runs at scale: you define the pipeline, run it, and get output. A workbench is optimized for iteration: you explore the data, try different filter thresholds, inspect what gets removed, search for specific content, find outliers, and gradually converge on the right curation strategy. Then you codify that strategy in a YAML config and run it in batch.

The technical choices follow from this philosophy:

**Streaming execution, constant memory.** Dokime processes documents one at a time through the filter chain. The `StreamingWriter` flushes to disk in configurable batches (default 10,000 documents). You can curate a 50GB JSONL file on a laptop with 8GB of RAM. Memory usage is determined by your batch size, not your dataset size.

**Per-filter removal stats.** Every pipeline run reports exactly how many documents each filter removed. This is not a nice-to-have -- it is the core feedback loop for data curation. If your repetition filter is removing 40% of documents, you probably need to raise the threshold. If your language filter is removing 2%, you probably do not need it. You cannot make these decisions without the numbers.

**HuggingFace Hub integration.** Load datasets directly from the Hub with streaming support. Push curated datasets back with one command (`dokime push`). The ML ecosystem runs on HuggingFace; your curation tool should too.

**Extensible filter architecture.** Every filter in Dokime inherits from a simple `Filter` abstract base class with one method: `filter(sample) -> bool`. Return `True` to keep, `False` to discard. Write your own filter in five lines of code and register it for YAML config support:

```python
from dokime.core.filters import Filter
from dokime.core.registry import register_filter
from dataclasses import dataclass

@dataclass
class ToxicityFilter(Filter):
    max_score: float = 0.5
    text_field: str = "text"

    def filter(self, sample):
        # your toxicity scoring logic here
        return compute_toxicity(sample[self.text_field]) <= self.max_score

    def name(self):
        return f"ToxicityFilter(max={self.max_score})"

register_filter("ToxicityFilter", ToxicityFilter)
```

Once registered, it works in YAML configs and gets included in per-filter removal stats just like any built-in filter.

**Data attribution.** This is the feature we are most excited about, and the one that no other open-source curation tool offers. More on this below.

## What's Next

### Data Attribution: Which Training Examples Help or Hurt Your Model?

Dokime already includes `dokime attribute`, which uses [TRAK](https://github.com/MadryLab/trak) to compute per-example influence scores. Given a training set and an evaluation set, it tells you which training examples improve model performance and which ones degrade it.

```bash
dokime attribute train.jsonl eval.jsonl \
  --model gpt2 --proj-dim 2048 --top 50
```

This produces a ranked list of the most harmful and most helpful training examples. The harmful ones are candidates for removal. The helpful ones are candidates for upweighting or finding more data like them.

Data attribution is computationally expensive and currently requires fine-tuning a proxy model. We are actively working on making it faster and more practical for large datasets. The current implementation works well for datasets up to ~50K examples on a single GPU. Scaling beyond that is an active research problem.

We are also exploring whether JEPA-style encoders can produce attribution-like signals more cheaply than gradient-based methods. This is speculative research, not a product promise -- but if it works, it would make data attribution practical at the scale where it matters most.

### Web UI

The `dokime explore` command currently launches a placeholder. We are building a FastAPI + web frontend for interactive exploration: browse documents, visualize filter effects, inspect embeddings in 2D projections, and drill into outliers. The goal is to make the "workbench" metaphor tangible -- you should be able to see your data, not just process it.

### Community

Dokime is Apache 2.0 licensed and we welcome contributions. The areas where we most need help:

- **More filters.** We have 16. Data-Juicer has 50. Every domain has its own quality signals -- toxicity detection, PII removal, code quality scoring, domain classifiers. If you have written a filter for your own project, consider contributing it.
- **Distributed execution.** We do not have Spark or Ray support yet. If you need to process petabyte-scale datasets, we would love help building this.
- **Web UI.** Frontend developers who want to help build the explore interface.
- **Benchmarks.** Rigorous comparisons of curation strategies and their effect on downstream model performance.

The test suite has 56 tests covering filters, pipelines, I/O, streaming, and quality scoring. We are aiming for high coverage on the core library and encourage tests with every PR.

### What We Are Not Building

To set expectations clearly: Dokime is not trying to replace Spark or Ray for petabyte-scale data processing. It is not a labeling tool (use Label Studio or Argilla for that). It is not an annotation platform. It is not a data versioning system (use DVC or lakeFS).

Dokime is a curation workbench. It helps you understand your data, decide what to keep, and produce a clean dataset for training. It is designed for datasets that fit on one machine (up to tens of millions of documents, roughly) and for workflows where a human is in the loop making decisions about quality thresholds. If your data fits that profile -- and most fine-tuning and domain-specific training datasets do -- Dokime is built for you.

---

## Get Started

```bash
pip install dokime          # core (filters, exact dedup, CLI)
pip install "dokime[all]"   # everything (embeddings, fuzzy dedup, language detection, HF datasets)
```

**Links:**

- **GitHub:** [github.com/dokime-ai/dokime](https://github.com/dokime-ai/dokime)
- **PyPI:** [pypi.org/project/dokime](https://pypi.org/project/dokime/)
- **Docs:** [dokime-ai.github.io/dokime](https://dokime-ai.github.io/dokime)

Star the repo if this is useful to you. Open an issue if something is broken. Open a PR if you want to make it better.

```bash
pip install dokime
```
