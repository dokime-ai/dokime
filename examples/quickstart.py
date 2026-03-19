"""
Dokime Quickstart
=================

A runnable example showing the full Dokime workflow:
  1. Load a dataset
  2. Filter with heuristic rules
  3. Deduplicate (exact + fuzzy)
  4. Compute embeddings
  5. Semantic search
  6. Find outliers

Run:
    pip install "dokime[all]"
    python examples/quickstart.py

This script creates a small synthetic dataset so it works out of the box.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def create_sample_dataset(path: Path) -> None:
    """Create a small synthetic JSONL dataset for demonstration."""
    samples = [
        {"text": "Machine learning models require large amounts of high-quality training data. "
                 "Data curation is the process of selecting, filtering, and cleaning raw data "
                 "to produce a dataset suitable for training."},
        {"text": "The transformer architecture has revolutionized natural language processing. "
                 "Attention mechanisms allow models to weigh the importance of different parts "
                 "of the input when producing each part of the output."},
        {"text": "Deduplication removes repeated or near-identical documents from a corpus. "
                 "This prevents models from memorizing specific examples and improves "
                 "generalization to unseen data."},
        {"text": "Embeddings map discrete tokens into continuous vector spaces where semantic "
                 "similarity corresponds to geometric proximity. Sentence embeddings capture "
                 "the meaning of entire passages."},
        {"text": "Data quality is more important than data quantity for most ML tasks. "
                 "A smaller, well-curated dataset often outperforms a larger, noisy one. "
                 "Filtering heuristics catch common quality issues."},
        {"text": "      "},  # junk: whitespace only
        {"text": "aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa"},  # junk: repetition
        {"text": "ok"},  # junk: too short
        {"text": "Machine learning models require large amounts of high-quality training data. "
                 "Data curation is the process of selecting, filtering, and cleaning raw data "
                 "to produce a dataset suitable for training."},  # exact duplicate of #0
        {"text": "Anomaly detection surfaces unusual documents that may be mislabeled, "
                 "corrupted, or otherwise different from the rest of the corpus. "
                 "Embedding-based methods compare each document to its neighbors."},
    ]
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Created sample dataset: {path} ({len(samples)} documents)")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ---------------------------------------------------------------
        # 1. Create sample data
        # ---------------------------------------------------------------
        input_path = tmpdir / "raw.jsonl"
        output_path = tmpdir / "curated.jsonl"
        create_sample_dataset(input_path)

        # ---------------------------------------------------------------
        # 2. Filter + deduplicate with a Pipeline
        # ---------------------------------------------------------------
        from dokime.core.filters import LengthFilter, RepetitionFilter, WhitespaceFilter
        from dokime.core.pipeline import Pipeline
        from dokime.quality.dedup import ExactDedup

        pipeline = Pipeline("quickstart")
        pipeline.add_filter(LengthFilter(min_length=50, max_length=100_000))
        pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=0.4))
        pipeline.add_filter(RepetitionFilter(max_repetition_ratio=0.3))
        pipeline.add_filter(ExactDedup())

        result = pipeline.run(str(input_path), str(output_path))
        print(f"\nKept {result['total_kept']} / {result['total_read']} documents")
        print(f"Removal rate: {result['removal_rate_pct']}%")

        # ---------------------------------------------------------------
        # 3. Compute embeddings
        # ---------------------------------------------------------------
        print("\n--- Computing embeddings ---")
        from dokime.embeddings.compute import compute_embeddings
        from dokime.io.readers import auto_read

        documents, embeddings = compute_embeddings(
            data=auto_read(str(output_path)),
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
        )
        print(f"Embeddings shape: {embeddings.shape}")

        # ---------------------------------------------------------------
        # 4. Semantic search
        # ---------------------------------------------------------------
        print("\n--- Semantic search ---")
        from dokime.embeddings.compute import EmbeddingModel
        from dokime.embeddings.search import EmbeddingIndex

        model = EmbeddingModel("all-MiniLM-L6-v2")
        index = EmbeddingIndex(embeddings, documents)

        query = "how to remove duplicate data"
        results = index.search(query, model, k=3)
        print(f"Query: '{query}'")
        for i, r in enumerate(results, 1):
            preview = r.document["text"][:80].replace("\n", " ")
            print(f"  {i}. [{r.score:.3f}] {preview}...")

        # ---------------------------------------------------------------
        # 5. Find outliers
        # ---------------------------------------------------------------
        print("\n--- Outlier detection ---")
        from dokime.embeddings.search import AnomalyScorer

        scorer = AnomalyScorer(embeddings)
        scores = scorer.score_all(k=3)
        print("Anomaly scores (higher = more unusual):")
        for i, (doc, score) in enumerate(zip(documents, scores)):
            preview = doc["text"][:60].replace("\n", " ")
            print(f"  [{score:.3f}] {preview}...")

        # ---------------------------------------------------------------
        # 6. Semantic dedup
        # ---------------------------------------------------------------
        print("\n--- Semantic deduplication ---")
        from dokime.embeddings.dedup import find_semantic_duplicates

        pairs = find_semantic_duplicates(embeddings, documents, threshold=0.90)
        if pairs:
            for a, b, sim in pairs:
                print(f"  Pair ({a}, {b}) similarity={sim:.3f}")
        else:
            print("  No semantic duplicates found above threshold.")

        print("\nDone.")


if __name__ == "__main__":
    main()
