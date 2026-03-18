"""Attribution engine — wraps TRAK to provide a simple API for data influence scoring.

This is the core of Dokime's differentiator: "which training examples help or hurt your model?"
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()


class AttributionEngine:
    """Compute data attribution scores for fine-tuning datasets using TRAK.

    Answers the question: "Which training examples improve or hurt
    model performance on a given evaluation set?"

    Example::

        engine = AttributionEngine(
            model_name="gpt2",
            train_data="finetune_data.jsonl",
            eval_data="eval_data.jsonl",
        )
        scores = engine.compute()
        # scores.shape = (n_eval, n_train)
        # Positive = training example helps on this eval example
        # Negative = training example hurts on this eval example

        harmful = engine.find_harmful(top_n=100)
        helpful = engine.find_helpful(top_n=100)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        train_data: str | list[dict[str, Any]] | None = None,
        eval_data: str | list[dict[str, Any]] | None = None,
        text_field: str = "text",
        max_length: int = 512,
        proj_dim: int = 2048,
        device: str | None = None,
        save_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.text_field = text_field
        self.max_length = max_length
        self.proj_dim = proj_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="dokime_attribution_")

        self._train_data = train_data
        self._eval_data = eval_data
        self._model = None
        self._tokenizer = None
        self._scores: np.ndarray | None = None

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Install attribution support: pip install dokime-ai[attribution]\n"
                "  (requires transformers, torch, traker)"
            ) from None

        console.print(f"[bold blue]Dokime[/] — Loading model [bold]{self.model_name}[/]")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        self._model = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
            self.model_name,
            attn_implementation="eager",  # Required for TRAK vmap compatibility
        )
        self._model.to(self.device)  # type: ignore[attr-defined]
        self._model.eval()  # type: ignore[attr-defined]

    def _load_data(self, data: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Load data from path or return list directly."""
        if isinstance(data, list):
            return data

        from dokime.io.readers import auto_read

        return list(auto_read(data))

    def _tokenize_dataset(self, documents: list[dict[str, Any]]) -> list[tuple[torch.Tensor, ...]]:
        """Tokenize documents into model-ready tensors."""
        assert self._tokenizer is not None

        batches = []
        for doc in documents:
            text = doc.get(self.text_field, "")
            tokens = self._tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
            # Labels are same as input_ids for causal LM, with padding masked to -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            batches.append((input_ids, attention_mask, labels))

        return batches

    def compute(self, quiet: bool = False) -> np.ndarray:
        """Compute attribution scores.

        Returns:
            Array of shape (n_eval, n_train) with per-example influence scores.
            Positive values indicate the training example helps on that eval example.
            Negative values indicate it hurts.
        """
        from trak import TRAKer

        from dokime.attribution.model_output import LanguageModelingModelOutput

        self._load_model()
        assert self._model is not None
        assert self._train_data is not None
        assert self._eval_data is not None

        train_docs = self._load_data(self._train_data)
        eval_docs = self._load_data(self._eval_data)

        if not quiet:
            console.print(f"  Train examples: {len(train_docs):,}")
            console.print(f"  Eval examples:  {len(eval_docs):,}")
            console.print(f"  Device: {self.device}")

        train_batches = self._tokenize_dataset(train_docs)
        eval_batches = self._tokenize_dataset(eval_docs)

        task = LanguageModelingModelOutput()

        traker = TRAKer(
            model=self._model,
            task=task,
            train_set_size=len(train_batches),
            save_dir=self.save_dir,
            load_from_save_dir=False,
            device=self.device,
            proj_dim=self.proj_dim,
            use_half_precision=False,  # float32 for stability
        )

        # Workaround for TRAK MmapSaver bug (v0.3.2, Windows): init_store creates
        # is_featurized with shape (n,), then load_current_store tries to re-create
        # it with shape (n, 1) in w+ mode. On Windows, the file is still locked by
        # the first mmap handle, causing OSError. Fix: skip init_store's is_featurized
        # creation and let load_current_store handle it entirely.
        _original_init_store = traker.saver.init_store

        def _patched_init_store(model_id: int) -> None:
            import os

            prefix = traker.saver.save_dir.joinpath(str(model_id))
            os.makedirs(prefix, exist_ok=True)
            # Skip creating _is_featurized.mmap here — let load_current_store do it
            traker.saver.load_current_store(model_id, mode="w+")

        traker.saver.init_store = _patched_init_store

        # Phase 1: Featurize training data
        if not quiet:
            console.print("\n[bold]Phase 1:[/] Featurizing training data...")

        traker.load_checkpoint(self._model.state_dict(), model_id=0)

        for _i, batch in enumerate(tqdm(train_batches, desc="Featurizing", disable=quiet)):
            batch_on_device = tuple(t.unsqueeze(0).to(self.device) for t in batch)
            traker.featurize(batch=batch_on_device, num_samples=1)

        traker.finalize_features()

        # Workaround: ensure experiments.json exists (TRAK bug on Windows)

        exp_file = Path(self.save_dir) / "experiments.json"
        if not exp_file.exists():
            exp_file.write_text("{}")

        # Phase 2: Score eval examples
        if not quiet:
            console.print("\n[bold]Phase 2:[/] Scoring eval examples...")

        traker.start_scoring_checkpoint(
            exp_name="dokime_attribution",
            checkpoint=self._model.state_dict(),
            model_id=0,
            num_targets=len(eval_batches),
        )

        for _i, batch in enumerate(tqdm(eval_batches, desc="Scoring", disable=quiet)):
            batch_on_device = tuple(t.unsqueeze(0).to(self.device) for t in batch)
            traker.score(batch=batch_on_device, num_samples=1)

        raw_scores = traker.finalize_scores(exp_name="dokime_attribution")
        self._scores = np.array(raw_scores)

        if not quiet:
            console.print(f"\n  Attribution scores shape: {self._scores.shape}")
            console.print(f"  Saved to: {self.save_dir}")

        return self._scores

    def aggregate_scores(self) -> np.ndarray:
        """Aggregate per-eval scores into a single per-training-example score.

        Returns:
            Array of shape (n_train,) — average influence of each training example
            across all eval examples. Higher = more helpful. Lower = more harmful.
        """
        if self._scores is None:
            raise RuntimeError("Call .compute() first.")

        return np.asarray(self._scores.mean(axis=0))

    def find_harmful(self, top_n: int = 50) -> list[tuple[int, float]]:
        """Find training examples that hurt model performance the most.

        Returns:
            List of (train_index, average_score) tuples, sorted most harmful first.
        """
        agg = self.aggregate_scores()
        indices = np.argsort(agg)[:top_n]
        return [(int(i), float(agg[i])) for i in indices]

    def find_helpful(self, top_n: int = 50) -> list[tuple[int, float]]:
        """Find training examples that help model performance the most.

        Returns:
            List of (train_index, average_score) tuples, sorted most helpful first.
        """
        agg = self.aggregate_scores()
        indices = np.argsort(agg)[-top_n:][::-1]
        return [(int(i), float(agg[i])) for i in indices]

    def summary(self) -> dict[str, Any]:
        """Get a summary of attribution results.

        Returns:
            Dict with counts of helpful/harmful examples and score statistics.
        """
        if self._scores is None:
            raise RuntimeError("Call .compute() first.")

        agg = self.aggregate_scores()
        return {
            "n_train": self._scores.shape[1],
            "n_eval": self._scores.shape[0],
            "n_helpful": int((agg > 0).sum()),
            "n_harmful": int((agg < 0).sum()),
            "n_neutral": int((agg == 0).sum()),
            "pct_harmful": round(float((agg < 0).sum()) / len(agg) * 100, 1),
            "mean_score": round(float(agg.mean()), 6),
            "std_score": round(float(agg.std()), 6),
            "min_score": round(float(agg.min()), 6),
            "max_score": round(float(agg.max()), 6),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of attribution results."""
        s = self.summary()

        table = Table(title="Attribution Summary", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Training examples", f"{s['n_train']:,}")
        table.add_row("Eval examples", f"{s['n_eval']:,}")
        table.add_row("Helpful examples", f"[green]{s['n_helpful']:,}[/]")
        table.add_row("Harmful examples", f"[red]{s['n_harmful']:,}[/]")
        table.add_row("% harmful", f"[red]{s['pct_harmful']}%[/]")
        table.add_row("Score range", f"{s['min_score']:.6f} to {s['max_score']:.6f}")

        console.print(table)
