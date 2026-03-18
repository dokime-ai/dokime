"""Dokime — HuggingFace Spaces demo.

A Gradio app showcasing Dokime's core data curation features:
filtering, quality scoring, and semantic search.
"""

from __future__ import annotations

import json
import time
from typing import Any

import gradio as gr

# ---------------------------------------------------------------------------
# Dokime imports
# ---------------------------------------------------------------------------
from dokime.core.filters import (
    AlphaFilter,
    LengthFilter,
    RepetitionFilter,
    WhitespaceFilter,
)
from dokime.core.pipeline import Pipeline
from dokime.quality.dedup import ExactDedup
from dokime.quality.scoring import QualityScorer

# ---------------------------------------------------------------------------
# Sample dataset — pre-loaded so users can click "Try it" immediately
# ---------------------------------------------------------------------------
SAMPLE_JSONL = r"""{"text": "Machine learning is a subfield of artificial intelligence that gives computers the ability to learn without being explicitly programmed. It focuses on the development of algorithms that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future."}
{"text": "The   quick    brown    fox   jumps    over   the   lazy    dog.   Too   much   whitespace   here.   Way   too   much.    Formatting   is   broken.    Lots    of    extra    spaces    everywhere    in    this    document.    Not    great    quality."}
{"text": "buy cheap viagra online best price discount pharmacy click here now limited time offer buy buy buy click click click spam spam spam buy cheap viagra online best price discount pharmacy click here now limited time offer buy buy buy click click click"}
{"text": "Deep learning architectures, particularly transformer models, have revolutionized natural language processing. Models like BERT, GPT, and T5 demonstrate that pre-training on large corpora followed by task-specific fine-tuning yields state-of-the-art results across a wide range of benchmarks. These advances have enabled practical applications from machine translation to code generation."}
{"text": "ab"}
{"text": "Data curation is the process of organizing, integrating, and maintaining data collections to ensure they meet the needs of downstream consumers. For machine learning, good data curation involves removing duplicates, filtering low-quality samples, balancing class distributions, and ensuring data provenance. Dokime provides tools for all of these tasks in a single Python package."}
{"text": "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon while the birds sing melodiously in the tall oak trees swaying gently in the warm summer breeze that carries the scent of wildflowers across the rolling green meadows stretching toward the distant purple mountains."}
{"text": "12345 67890 !@#$% ^&*() 98765 43210 !@#$% ^&*() 12345 67890 !@#$% ^&*() numbers and symbols only 54321 09876 !@#$% ^&*()"}
{"text": "Data curation is the process of organizing, integrating, and maintaining data collections to ensure they meet the needs of downstream consumers. For machine learning, good data curation involves removing duplicates, filtering low-quality samples, balancing class distributions, and ensuring data provenance. Dokime provides tools for all of these tasks in a single Python package."}
{"text": "Retrieval-augmented generation (RAG) combines information retrieval with language generation to produce more factual and grounded outputs. By retrieving relevant passages from a knowledge base before generating a response, RAG systems can reduce hallucination and provide up-to-date information that was not seen during pre-training. This approach is becoming standard in enterprise AI deployments."}""".strip()


# ---------------------------------------------------------------------------
# Helper: parse JSONL input (tolerant)
# ---------------------------------------------------------------------------
def _parse_jsonl(raw: str) -> list[dict[str, Any]]:
    """Parse JSONL text into a list of dicts. Skips blank / unparseable lines."""
    docs: list[dict[str, Any]] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                docs.append(obj)
        except json.JSONDecodeError:
            # Treat bare text lines as {"text": line}
            docs.append({"text": line})
    return docs


def _docs_to_jsonl(docs: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(d, ensure_ascii=False) for d in docs)


# ===================================================================
# Tab 1: Curate
# ===================================================================
def run_curate(
    raw_jsonl: str,
    min_length: int,
    max_whitespace: float,
    max_repetition: float,
    enable_dedup: bool,
) -> tuple[str, str]:
    """Build and run a Dokime pipeline on the pasted JSONL data."""
    docs = _parse_jsonl(raw_jsonl)
    if not docs:
        return "", "No valid documents found in input."

    # Build pipeline
    pipeline = Pipeline("spaces-demo")
    pipeline.add_filter(LengthFilter(min_length=min_length))
    pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=max_whitespace))
    pipeline.add_filter(RepetitionFilter(max_repetition_ratio=max_repetition))
    if enable_dedup:
        pipeline.add_filter(ExactDedup())

    # Run filter chain manually to collect per-filter stats
    filter_stats: dict[str, int] = {f.name(): 0 for f in pipeline.filters}
    kept: list[dict[str, Any]] = []
    start = time.time()

    for sample in docs:
        passed = True
        for f in pipeline.filters:
            if not f.filter(sample):
                filter_stats[f.name()] += 1
                passed = False
                break
        if passed:
            kept.append(sample)

    elapsed = time.time() - start
    total_read = len(docs)
    total_kept = len(kept)
    total_removed = total_read - total_kept
    removal_pct = (total_removed / total_read * 100) if total_read else 0

    # Build stats markdown table
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Documents read | {total_read} |",
        f"| Documents kept | {total_kept} |",
        f"| Documents removed | {total_removed} |",
        f"| Removal rate | {removal_pct:.1f}% |",
        f"| Elapsed | {elapsed:.3f}s |",
        "",
        "**Per-filter removals:**",
        "",
        "| Filter | Removed |",
        "|---|---|",
    ]
    for fname, count in filter_stats.items():
        lines.append(f"| `{fname}` | {count} |")

    return _docs_to_jsonl(kept), "\n".join(lines)


# ===================================================================
# Tab 2: Score
# ===================================================================
def run_score(raw_text: str) -> str:
    """Score each document with Dokime's QualityScorer."""
    docs = _parse_jsonl(raw_text)
    if not docs:
        return "No valid documents found in input."

    scorer = QualityScorer()

    header = (
        "| # | text (first 80 chars) | _char_entropy | _alpha_ratio | "
        "_whitespace_ratio | _special_ratio | _avg_word_length | _quality_score |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for i, doc in enumerate(docs, 1):
        scored = scorer.score(doc)
        preview = scored.get("text", "")[:80].replace("|", "\\|").replace("\n", " ")
        rows.append(
            f"| {i} | {preview} | {scored['_char_entropy']} | "
            f"{scored['_alpha_ratio']} | {scored['_whitespace_ratio']} | "
            f"{scored['_special_ratio']} | {scored['_avg_word_length']} | "
            f"**{scored['_quality_score']}** |"
        )

    return "\n".join(rows)


# ===================================================================
# Tab 3: Search
# ===================================================================

# Lazy-loaded globals for search (avoid loading model on startup)
_search_model = None
_search_index = None
_search_docs: list[dict[str, Any]] = []


def _ensure_search_model():
    global _search_model
    if _search_model is None:
        from dokime.embeddings.compute import EmbeddingModel

        _search_model = EmbeddingModel("all-MiniLM-L6-v2")
    return _search_model


def run_search(raw_jsonl: str, query: str, top_k: int) -> str:
    """Embed the dataset, build a FAISS index, and search."""
    global _search_index, _search_docs

    docs = _parse_jsonl(raw_jsonl)
    if not docs:
        return "No valid documents found in input."
    if not query.strip():
        return "Please enter a search query."

    model = _ensure_search_model()

    # Recompute index if the docs changed
    texts = [d.get("text", "") for d in docs]
    import numpy as np

    embeddings = model.encode_texts(texts, show_progress=False)

    from dokime.embeddings.search import EmbeddingIndex

    index = EmbeddingIndex(embeddings, docs)
    results = index.search(query, model, k=min(top_k, len(docs)))

    if not results:
        return "No results found."

    header = "| Rank | Score | Text (first 120 chars) |"
    sep = "|---|---|---|"
    rows = [header, sep]
    for rank, r in enumerate(results, 1):
        preview = r.document.get("text", "")[:120].replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {rank} | {r.score:.4f} | {preview} |")

    return "\n".join(rows)


# ===================================================================
# UI — dark theme, polished layout
# ===================================================================

DESCRIPTION = """\
# Dokime

**The open-source workbench for ML training data.**

Filter junk, deduplicate, score quality, compute embeddings, and search your \
dataset -- all from one `pip install`.
"""

CSS = """
.dokime-header {
    text-align: center;
    margin-bottom: 0.5em;
}
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
    title="Dokime — Data Curation Toolkit",
) as demo:

    gr.Markdown(DESCRIPTION, elem_classes=["dokime-header"])

    # ----- Tab 1: Curate -----
    with gr.Tab("Curate"):
        gr.Markdown(
            "Paste JSONL data below (each line: `{\"text\": \"...\"}`) and "
            "configure filters. Click **Run Pipeline** to see what Dokime keeps "
            "and what it removes."
        )
        with gr.Row():
            with gr.Column(scale=2):
                curate_input = gr.Textbox(
                    label="Input (JSONL)",
                    lines=12,
                    max_lines=30,
                    placeholder='{"text": "Your document here..."}\n{"text": "Another document..."}',
                )
                curate_sample_btn = gr.Button(
                    "Load sample dataset", variant="secondary", size="sm"
                )
            with gr.Column(scale=1):
                min_length = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=20,
                    step=5,
                    label="Min length (chars)",
                )
                max_whitespace = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Max whitespace ratio",
                )
                max_repetition = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Max repetition ratio",
                )
                enable_dedup = gr.Checkbox(label="Exact deduplication", value=True)
                curate_btn = gr.Button("Run Pipeline", variant="primary")

        curate_stats = gr.Markdown(label="Statistics")
        curate_output = gr.Textbox(
            label="Output (kept documents, JSONL)", lines=10, max_lines=30
        )

        curate_sample_btn.click(
            fn=lambda: SAMPLE_JSONL, outputs=[curate_input]
        )
        curate_btn.click(
            fn=run_curate,
            inputs=[curate_input, min_length, max_whitespace, max_repetition, enable_dedup],
            outputs=[curate_output, curate_stats],
        )

    # ----- Tab 2: Score -----
    with gr.Tab("Score"):
        gr.Markdown(
            "Paste documents below (JSONL or one text per line) and click "
            "**Score Quality** to see per-document quality signals."
        )
        score_input = gr.Textbox(
            label="Input (JSONL or plain text, one doc per line)",
            lines=10,
            max_lines=30,
            placeholder='{"text": "Your document here..."}\n{"text": "Another document..."}',
        )
        with gr.Row():
            score_sample_btn = gr.Button(
                "Load sample dataset", variant="secondary", size="sm"
            )
            score_btn = gr.Button("Score Quality", variant="primary")
        score_output = gr.Markdown(label="Quality Scores")

        score_sample_btn.click(fn=lambda: SAMPLE_JSONL, outputs=[score_input])
        score_btn.click(fn=run_score, inputs=[score_input], outputs=[score_output])

    # ----- Tab 3: Search -----
    with gr.Tab("Search"):
        gr.Markdown(
            "Paste a JSONL dataset, enter a natural-language query, and "
            "Dokime will rank documents by semantic similarity using "
            "sentence-transformers + FAISS.\n\n"
            "> **Note:** The first search loads the `all-MiniLM-L6-v2` model "
            "(~80 MB). This may take 10-30 seconds on a cold start."
        )
        search_input = gr.Textbox(
            label="Dataset (JSONL)",
            lines=10,
            max_lines=30,
            placeholder='{"text": "Your document here..."}\n{"text": "Another document..."}',
        )
        with gr.Row():
            search_query = gr.Textbox(
                label="Search query",
                placeholder="e.g. retrieval augmented generation",
                scale=3,
            )
            search_k = gr.Slider(
                minimum=1, maximum=20, value=5, step=1, label="Top K", scale=1
            )
        with gr.Row():
            search_sample_btn = gr.Button(
                "Load sample dataset", variant="secondary", size="sm"
            )
            search_btn = gr.Button("Search", variant="primary")
        search_output = gr.Markdown(label="Search Results")

        search_sample_btn.click(fn=lambda: SAMPLE_JSONL, outputs=[search_input])
        search_btn.click(
            fn=run_search,
            inputs=[search_input, search_query, search_k],
            outputs=[search_output],
        )

    # ----- Tab 4: About -----
    with gr.Tab("About"):
        gr.Markdown(
            """
## What is Dokime?

Dokime is an open-source Python toolkit for curating ML training data. It
provides heuristic filters, deduplication (exact, fuzzy, and semantic),
quality scoring, embedding computation, semantic search, and outlier
detection -- all from a single `pip install`.

### Core capabilities

- **12 built-in heuristic filters** -- length, whitespace, repetition, special
  characters, alpha ratio, URLs, stopwords, language, regex, and more
- **3 deduplication strategies** -- exact (SHA-256), fuzzy (MinHash-LSH), and
  semantic (embedding cosine similarity)
- **Quality scoring** -- per-document signals including character entropy,
  alpha ratio, composite quality score
- **Embeddings & search** -- sentence-transformer embeddings, FAISS
  nearest-neighbor search, anomaly/outlier detection
- **Pipeline orchestration** -- chain filters in code or YAML, get per-filter
  removal stats
- **CLI + Python SDK** -- seven commands, zero boilerplate

### Links

| | |
|---|---|
| GitHub | [github.com/dokime-ai/dokime](https://github.com/dokime-ai/dokime) |
| PyPI | [pypi.org/project/dokime-ai](https://pypi.org/project/dokime-ai/) |
| Docs | [dokime-ai.github.io/dokime](https://dokime-ai.github.io/dokime) |

### Install

```bash
pip install dokime-ai
```

For the full toolkit (embeddings, fuzzy dedup, language detection):

```bash
pip install "dokime-ai[all]"
```

### License

Apache 2.0
"""
        )

if __name__ == "__main__":
    demo.launch()
