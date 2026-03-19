"""Dokime CLI — data curation from the command line."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from dokime.core.pipeline import Pipeline

app = typer.Typer(
    name="dokime",
    help="Open-source data curation toolkit for ML.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the current version."""
    from dokime import __version__

    typer.echo(f"dokime {__version__}")


@app.command()
def curate(
    input_path: str = typer.Argument(
        ..., help="Path to input dataset (JSONL, Parquet, CSV, or HuggingFace dataset ID)"
    ),
    output_path: str = typer.Argument(..., help="Path to write curated output"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML pipeline config"),
    min_length: int | None = typer.Option(None, help="Min document length (chars)"),
    max_length: int | None = typer.Option(None, help="Max document length (chars)"),
    max_whitespace: float | None = typer.Option(None, help="Max whitespace ratio (0-1)"),
    max_repetition: float | None = typer.Option(None, help="Max n-gram repetition ratio (0-1)"),
    max_special: float | None = typer.Option(None, help="Max special character ratio (0-1)"),
    dedup: bool = typer.Option(False, "--dedup", help="Enable exact deduplication"),
    fuzzy_dedup: float | None = typer.Option(None, "--fuzzy-dedup", help="MinHash dedup threshold (0-1)"),
) -> None:
    """Run a curation pipeline on a dataset."""
    from dokime.core.pipeline import Pipeline

    if config:
        pipeline = Pipeline.from_config(config)
    else:
        pipeline = _build_pipeline_from_flags(
            min_length=min_length,
            max_length=max_length,
            max_whitespace=max_whitespace,
            max_repetition=max_repetition,
            max_special=max_special,
            dedup=dedup,
            fuzzy_dedup=fuzzy_dedup,
        )

    if not pipeline.filters:
        typer.echo("No filters specified. Use --config or filter flags (--min-length, --dedup, etc.)")
        typer.echo("Run 'dokime curate --help' for options.")
        raise typer.Exit(1)

    pipeline.run(input_path, output_path)


@app.command()
def stats(
    input_path: str = typer.Argument(..., help="Path to dataset to analyze"),
    text_field: str = typer.Option("text", "--field", help="Field containing text"),
) -> None:
    """Show basic statistics about a dataset."""
    from rich.console import Console
    from rich.table import Table

    from dokime.io.readers import auto_read

    console = Console()
    console.print(f"[bold blue]Dokime[/] — Analyzing [bold]{input_path}[/]")

    total = 0
    total_chars = 0
    min_len = float("inf")
    max_len = 0

    for sample in auto_read(input_path):
        total += 1
        text = sample.get(text_field, "")
        length = len(text)
        total_chars += length
        min_len = min(min_len, length)
        max_len = max(max_len, length)

    if total == 0:
        console.print("[yellow]Empty dataset.[/]")
        raise typer.Exit()

    avg_len = total_chars / total

    table = Table(title="Dataset Statistics", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total documents", f"{total:,}")
    table.add_row("Total characters", f"{total_chars:,}")
    table.add_row("Avg length", f"{avg_len:,.0f}")
    table.add_row("Min length", f"{int(min_len):,}")
    table.add_row("Max length", f"{max_len:,}")
    console.print(table)


@app.command()
def embed(
    input_path: str = typer.Argument(..., help="Path to input dataset"),
    output_path: str = typer.Argument(..., help="Path to save embeddings (.npy)"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Sentence-transformer model name"),
    text_field: str = typer.Option("text", "--field", help="Field containing text"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for encoding"),
    device: str | None = typer.Option(None, "--device", help="Device (cpu, cuda, etc.)"),
) -> None:
    """Compute embeddings for a dataset."""
    from dokime.embeddings.compute import compute_embeddings
    from dokime.io.readers import auto_read

    compute_embeddings(
        data=auto_read(input_path),
        model_name=model,
        text_field=text_field,
        batch_size=batch_size,
        output_path=output_path,
        device=device,
    )


@app.command()
def search(
    input_path: str = typer.Argument(..., help="Path to dataset"),
    query: str = typer.Argument(..., help="Search query"),
    embeddings_path: str | None = typer.Option(None, "--embeddings", "-e", help="Path to precomputed .npy embeddings"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Sentence-transformer model name"),
    text_field: str = typer.Option("text", "--field", help="Field containing text"),
    k: int = typer.Option(5, "-k", help="Number of results"),
) -> None:
    """Semantic search over a dataset."""
    import numpy as np
    from rich.console import Console

    from dokime.embeddings.compute import EmbeddingModel, compute_embeddings
    from dokime.embeddings.search import EmbeddingIndex
    from dokime.io.readers import auto_read

    console = Console()

    if embeddings_path:
        documents = list(auto_read(input_path))
        embeddings = np.load(embeddings_path)
    else:
        documents, embeddings = compute_embeddings(
            data=auto_read(input_path),
            model_name=model,
            text_field=text_field,
        )

    emb_model = EmbeddingModel(model)
    index = EmbeddingIndex(embeddings, documents)
    results = index.search(query, emb_model, k=k)

    console.print(f"\n[bold blue]Dokime[/] — Top {k} results for: [bold]{query}[/]\n")
    for i, r in enumerate(results, 1):
        text_preview = r.document.get(text_field, "")[:120].replace("\n", " ")
        console.print(f"  [bold]{i}.[/] [green]{r.score:.4f}[/]  {text_preview}")
    console.print()


@app.command()
def outliers(
    input_path: str = typer.Argument(..., help="Path to dataset"),
    embeddings_path: str | None = typer.Option(None, "--embeddings", "-e", help="Path to precomputed .npy embeddings"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Sentence-transformer model name"),
    text_field: str = typer.Option("text", "--field", help="Field containing text"),
    k: int = typer.Option(10, "-k", help="Number of neighbors for scoring"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of outliers to show"),
) -> None:
    """Find anomalous/outlier documents in a dataset."""
    import numpy as np
    from rich.console import Console

    from dokime.embeddings.compute import compute_embeddings
    from dokime.embeddings.search import AnomalyScorer
    from dokime.io.readers import auto_read

    console = Console()

    if embeddings_path:
        documents = list(auto_read(input_path))
        embeddings = np.load(embeddings_path)
    else:
        documents, embeddings = compute_embeddings(
            data=auto_read(input_path),
            model_name=model,
            text_field=text_field,
        )

    scorer = AnomalyScorer(embeddings)
    outlier_indices = scorer.find_outliers(k=k, top_n=top_n)
    scores = scorer.score_all(k=k)

    console.print(f"\n[bold blue]Dokime[/] — Top {top_n} outliers\n")
    for rank, idx in enumerate(outlier_indices, 1):
        text_preview = documents[idx].get(text_field, "")[:120].replace("\n", " ")
        console.print(f"  [bold]{rank}.[/] [red]{scores[idx]:.4f}[/]  {text_preview}")
    console.print()


@app.command()
def score(
    input_path: str = typer.Argument(..., help="Path to dataset"),
    output_path: str | None = typer.Argument(None, help="Optional: save scored output (JSONL or Parquet)"),
    text_field: str = typer.Option("text", "--field", help="Field containing text"),
    worst: int = typer.Option(5, "--worst", "-w", help="Number of worst documents to show"),
) -> None:
    """Score your training data quality. Get a grade in 10 seconds."""
    from dokime.quality.report import run_report

    run_report(input_path, text_field=text_field, show_worst=worst)

    # Optionally save scored output
    if output_path is not None:
        from dokime.io.readers import auto_read
        from dokime.io.writers import StreamingWriter
        from dokime.quality.scoring import QualityScorer

        scorer = QualityScorer(text_field=text_field)
        writer = StreamingWriter(output_path)
        count = 0
        for doc in auto_read(input_path):
            writer.write(scorer.score(doc))
            count += 1
        writer.close()
        typer.echo(f"Scored {count:,} documents -> {output_path}")


@app.command()
def diagnose(
    input_path: str = typer.Argument(..., help="Path to dataset to diagnose"),
    text_field: str = typer.Option("text", "--field", help="Field containing text"),
    embeddings_path: str | None = typer.Option(None, "--embeddings", "-e", help="Path to precomputed .npy embeddings"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Sentence-transformer model for embeddings"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for embedding computation"),
    device: str | None = typer.Option(None, "--device", help="Device (cpu, cuda, etc.)"),
    skip_embeddings: bool = typer.Option(False, "--skip-embeddings", help="Skip embedding-based analyses"),
    minhash_threshold: float = typer.Option(0.8, "--minhash-threshold", help="MinHash similarity threshold"),
    semantic_threshold: float = typer.Option(0.95, "--semantic-threshold", help="Semantic dedup threshold"),
    worst: int = typer.Option(5, "--worst", "-w", help="Number of worst documents to show"),
    outlier_count: int = typer.Option(5, "--outliers", "-o", help="Number of outliers to show"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Full health check: quality + duplicates + outliers in one command."""
    from dokime.quality.diagnose import run_diagnose

    result = run_diagnose(
        input_path=input_path,
        text_field=text_field,
        embeddings_path=embeddings_path,
        model=model,
        batch_size=batch_size,
        device=device,
        minhash_threshold=minhash_threshold,
        semantic_threshold=semantic_threshold,
        skip_embeddings=skip_embeddings,
        show_worst=worst,
        show_outliers=outlier_count,
        quiet=json_output,
    )

    if json_output:
        import json as json_mod
        from dataclasses import asdict

        typer.echo(json_mod.dumps(asdict(result), indent=2, default=str))


@app.command()
def push(
    input_path: str = typer.Argument(..., help="Path to dataset to push"),
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g., username/my-dataset)"),
    split: str = typer.Option("train", "--split", help="Dataset split name"),
    private: bool = typer.Option(False, "--private", help="Make the dataset private"),
    token: str | None = typer.Option(None, "--token", help="HuggingFace API token"),
) -> None:
    """Push a dataset to HuggingFace Hub."""
    from dokime.io.hub import push_to_hub
    from dokime.io.readers import auto_read

    push_to_hub(
        data=auto_read(input_path),
        repo_id=repo_id,
        split=split,
        private=private,
        token=token,
    )


@app.command()
def attribute(
    train_path: str = typer.Argument(..., help="Path to training dataset"),
    eval_path: str = typer.Argument(..., help="Path to evaluation dataset"),
    model: str = typer.Option("gpt2", "--model", "-m", help="HuggingFace model name for proxy"),
    text_field: str = typer.Option("text", "--field", help="Field containing text"),
    max_length: int = typer.Option(512, "--max-length", help="Max token length"),
    proj_dim: int = typer.Option(2048, "--proj-dim", help="TRAK projection dimension"),
    top_n: int = typer.Option(20, "--top", "-n", help="Number of harmful/helpful examples to show"),
    save_dir: str | None = typer.Option(None, "--save-dir", help="Directory to save attribution results"),
    device: str | None = typer.Option(None, "--device", help="Device (cpu, cuda)"),
) -> None:
    """Find training examples that help or hurt model performance.

    Uses TRAK-based data attribution to score each training example's
    influence on evaluation performance. Requires: pip install dokime[attribution]
    """
    try:
        from dokime.attribution.engine import AttributionEngine
    except ImportError:
        typer.echo("Install attribution support: pip install dokime[attribution]")
        raise typer.Exit(1) from None

    from rich.console import Console

    console = Console()

    engine = AttributionEngine(
        model_name=model,
        train_data=train_path,
        eval_data=eval_path,
        text_field=text_field,
        max_length=max_length,
        proj_dim=proj_dim,
        device=device,
        save_dir=save_dir,
    )

    engine.compute()
    engine.print_summary()

    # Show top harmful
    harmful = engine.find_harmful(top_n=top_n)
    console.print(f"\n[bold red]Top {top_n} harmful training examples:[/]\n")
    train_docs = engine._load_data(train_path)
    for rank, (idx, score) in enumerate(harmful, 1):
        text = train_docs[idx].get(text_field, "")[:100].replace("\n", " ")
        console.print(f"  {rank}. [{score:+.6f}] {text}")

    # Show top helpful
    helpful = engine.find_helpful(top_n=top_n)
    console.print(f"\n[bold green]Top {top_n} helpful training examples:[/]\n")
    for rank, (idx, score) in enumerate(helpful, 1):
        text = train_docs[idx].get(text_field, "")[:100].replace("\n", " ")
        console.print(f"  {rank}. [{score:+.6f}] {text}")


@app.command("eval-physics")
def eval_physics(
    model_url: str = typer.Option(..., "--model-url", help="OpenAI-compatible API endpoint (chat/completions URL)"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key (Bearer token)"),
    dataset: str | None = typer.Option(None, "--dataset", help="Path to PhyX_MC.tsv (uses bundled default if omitted)"),
    limit: int = typer.Option(0, "--limit", help="Limit number of questions (0=all)"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Evaluate a vision-language model on physics reasoning (PhyX benchmark).

    Sends each PhyX multiple-choice question (image + text) to the model,
    scores answers with rule-based string matching, and reports accuracy
    broken down by physics domain.

    Example:
        dokime eval-physics --model-url http://localhost:8000/v1/chat/completions --limit 10
    """
    from dokime.eval.physics import print_report, run_evaluation

    if not json_output:
        from rich.console import Console

        console = Console()
        console.print("[bold blue]Dokime[/] eval-physics starting...")

        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task("Evaluating...", total=None)

            def on_progress(current: int, total: int, qr: object) -> None:
                progress.update(task, total=total, completed=current)

            result = run_evaluation(
                model_url=model_url,
                api_key=api_key,
                dataset_path=dataset,
                limit=limit,
                on_progress=on_progress,
            )

        print_report(result)
    else:
        import json as json_mod

        result = run_evaluation(
            model_url=model_url,
            api_key=api_key,
            dataset_path=dataset,
            limit=limit,
        )
        typer.echo(json_mod.dumps(result.to_dict(), indent=2))


@app.command()
def explore(
    input_path: str = typer.Argument(..., help="Path to dataset to explore"),
    port: int = typer.Option(8765, "--port", "-p", help="Port for the web UI"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    embeddings: str | None = typer.Option(None, "--embeddings", "-e", help="Path to precomputed .npy embeddings"),
) -> None:
    """Launch the interactive web UI to explore a dataset."""
    try:
        from dokime.explore.server import launch
    except ImportError:
        typer.echo("Install explore dependencies: pip install dokime[explore]")
        raise typer.Exit(1) from None

    launch(input_path=input_path, host=host, port=port, embeddings_path=embeddings)


def _build_pipeline_from_flags(
    min_length: int | None,
    max_length: int | None,
    max_whitespace: float | None,
    max_repetition: float | None,
    max_special: float | None,
    dedup: bool,
    fuzzy_dedup: float | None,
) -> Pipeline:
    """Build a pipeline from CLI flags."""
    from dokime.core.filters import LengthFilter, RepetitionFilter, SpecialCharFilter, WhitespaceFilter
    from dokime.core.pipeline import Pipeline

    pipeline = Pipeline("cli")

    if min_length is not None or max_length is not None:
        pipeline.add_filter(
            LengthFilter(
                min_length=min_length or 0,
                max_length=max_length or 1_000_000,
            )
        )

    if max_whitespace is not None:
        pipeline.add_filter(WhitespaceFilter(max_whitespace_ratio=max_whitespace))

    if max_repetition is not None:
        pipeline.add_filter(RepetitionFilter(max_repetition_ratio=max_repetition))

    if max_special is not None:
        pipeline.add_filter(SpecialCharFilter(max_special_ratio=max_special))

    if dedup:
        from dokime.quality.dedup import ExactDedup

        pipeline.add_filter(ExactDedup())

    if fuzzy_dedup is not None:
        from dokime.quality.dedup import MinHashDedup

        pipeline.add_filter(MinHashDedup(threshold=fuzzy_dedup))

    return pipeline


if __name__ == "__main__":
    app()
