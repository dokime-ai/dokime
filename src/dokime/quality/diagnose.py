"""Full dataset health check — the comprehensive diagnostic tool.

`dokime diagnose data.jsonl` runs quality scoring, deduplication detection,
outlier detection, and semantic analysis in one command, producing an
actionable terminal report.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from dokime.quality.report import _bar, _grade_letter
from dokime.quality.scoring import QualityScorer

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class DiagnoseResult:
    """Structured container for all diagnostic findings."""

    # Metadata
    input_path: str
    total_documents: int
    elapsed_seconds: float

    # Quality (always available)
    grade: str
    grade_color: str
    overall_score: float
    quality_distribution: dict[str, int]
    issue_counts: dict[str, int]
    worst_documents: list[tuple[int, float]]

    # Exact dedup (always available)
    exact_duplicate_count: int

    # MinHash dedup (None if deps not installed)
    minhash_available: bool = False
    minhash_duplicate_count: int | None = None

    # Embeddings-based (None if deps not installed)
    embeddings_available: bool = False
    outlier_count: int | None = None
    outlier_indices: list[int] = field(default_factory=list)
    outlier_scores: list[float] = field(default_factory=list)
    semantic_duplicate_count: int | None = None
    semantic_duplicate_pairs: list[tuple[int, int, float]] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)


def run_diagnose(
    input_path: str,
    text_field: str = "text",
    embeddings_path: str | None = None,
    model: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str | None = None,
    minhash_threshold: float = 0.8,
    semantic_threshold: float = 0.95,
    outlier_k: int = 10,
    show_worst: int = 5,
    show_outliers: int = 5,
    skip_embeddings: bool = False,
    quiet: bool = False,
) -> DiagnoseResult:
    """Run a full diagnostic on a dataset.

    Composes quality scoring, deduplication, and outlier detection into a
    single report. Gracefully degrades when optional dependencies are missing.

    Returns:
        DiagnoseResult with all findings.
    """
    from dokime.io.readers import auto_read

    start = time.time()

    if not quiet:
        console.print(f"\n[bold blue]Dokime[/] — Diagnosing [bold]{input_path}[/] ...\n")

    # 1. Materialize dataset (needed for embeddings, dedup, random access)
    documents = list(auto_read(input_path))
    total = len(documents)

    if total == 0:
        if not quiet:
            console.print("[yellow]Empty dataset. Nothing to diagnose.[/]")
        return DiagnoseResult(
            input_path=input_path,
            total_documents=0,
            elapsed_seconds=0.0,
            grade="N/A",
            grade_color="dim",
            overall_score=0.0,
            quality_distribution={"excellent": 0, "good": 0, "medium": 0, "poor": 0},
            issue_counts={},
            worst_documents=[],
            exact_duplicate_count=0,
        )

    # 2. Quality scoring + exact dedup in one pass
    scorer = QualityScorer(text_field=text_field)
    scored_docs: list[dict[str, Any]] = []
    scores: list[float] = []
    hashes: set[str] = set()
    exact_dup_count = 0

    for doc in documents:
        scored = scorer.score(doc)
        scored_docs.append(scored)
        scores.append(scored["_quality_score"])

        text = doc.get(text_field, "")
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if h in hashes:
            exact_dup_count += 1
        else:
            hashes.add(h)

    avg_score = sum(scores) / total
    overall_pct = avg_score * 100
    grade, grade_color = _grade_letter(overall_pct)

    quality_dist = {
        "excellent": sum(1 for s in scores if s >= 0.8),
        "good": sum(1 for s in scores if 0.6 <= s < 0.8),
        "medium": sum(1 for s in scores if 0.3 <= s < 0.6),
        "poor": sum(1 for s in scores if s < 0.3),
    }

    issue_counts = {
        "short_docs": sum(1 for d in scored_docs if d["_word_count"] < 50),
        "low_quality": sum(1 for s in scores if s < 0.3),
        "low_entropy": sum(1 for d in scored_docs if d.get("_char_entropy", 5) < 2.0),
        "high_special": sum(1 for d in scored_docs if d.get("_special_ratio", 0) > 0.3),
        # New: Gopher/C4/FineWeb signals
        "high_repetition": sum(1 for d in scored_docs if d.get("_dup_5gram_frac", 0) > 0.15),
        "low_punctuation": sum(
            1 for d in scored_docs if d.get("_line_punct_ratio", 1) < 0.12 and d["_word_count"] >= 50
        ),
        "high_dup_lines": sum(1 for d in scored_docs if d.get("_dup_line_frac", 0) > 0.30),
        "boilerplate": sum(1 for d in scored_docs if d.get("_boilerplate_lines", 0) > 0),
        "few_sentences": sum(1 for d in scored_docs if d.get("_sentence_count", 99) < 3 and d["_word_count"] >= 50),
    }

    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])
    worst = [(i, s) for i, s in indexed_scores[:show_worst]]

    if not quiet:
        console.print(f"  [green]Quality scoring complete[/] ({total:,} docs)")

    # 3. MinHash dedup (optional)
    minhash_available = False
    minhash_dup_count = None
    try:
        from dokime.quality.dedup import MinHashDedup

        mhd = MinHashDedup(threshold=minhash_threshold, text_field=text_field)
        minhash_dup_count = sum(1 for doc in documents if not mhd.filter(doc))
        minhash_available = True
        if not quiet:
            console.print(f"  [green]MinHash dedup complete[/] ({minhash_dup_count:,} near-dupes)")
    except ImportError:
        if not quiet:
            console.print("  [yellow]MinHash dedup skipped[/] (install dokime[dedup])")

    # 4. Embedding-based analyses (optional)
    embeddings_available = False
    outlier_count = None
    outlier_idx: list[int] = []
    outlier_sc: list[float] = []
    sem_dup_count = None
    sem_dup_pairs: list[tuple[int, int, float]] = []

    if not skip_embeddings:
        try:
            import numpy as np

            if embeddings_path:
                embeddings = np.load(embeddings_path)
                if not quiet:
                    console.print(f"  [green]Loaded embeddings[/] from {embeddings_path}")
            else:
                from dokime.embeddings.compute import compute_embeddings

                _, embeddings = compute_embeddings(
                    data=documents,
                    model_name=model,
                    text_field=text_field,
                    batch_size=batch_size,
                    device=device,
                    quiet=quiet,
                )

            embeddings_available = True

            # Outlier detection
            from dokime.embeddings.search import AnomalyScorer

            anom_scorer = AnomalyScorer(embeddings)
            all_anomaly_scores = anom_scorer.score_all(k=outlier_k)
            outlier_idx_raw = anom_scorer.find_outliers(k=outlier_k)
            outlier_count = len(outlier_idx_raw)
            outlier_idx = outlier_idx_raw[:show_outliers]
            outlier_sc = [float(all_anomaly_scores[i]) for i in outlier_idx]
            if not quiet:
                console.print(f"  [green]Outlier detection complete[/] ({outlier_count:,} outliers)")

            # Semantic dedup
            from dokime.embeddings.dedup import find_semantic_duplicates

            sem_dup_pairs = find_semantic_duplicates(embeddings, documents, threshold=semantic_threshold, quiet=True)
            sem_dup_count = len(sem_dup_pairs)
            if not quiet:
                console.print(f"  [green]Semantic dedup complete[/] ({sem_dup_count:,} pairs)")

        except ImportError:
            if not quiet:
                console.print("  [yellow]Embedding analyses skipped[/] (install dokime[embeddings])")
    elif not quiet:
        console.print("  [dim]Embedding analyses skipped[/] (--skip-embeddings)")

    elapsed = time.time() - start

    # 5. Build recommendations
    recs = _build_recommendations(
        input_path=input_path,
        total=total,
        exact_dup_count=exact_dup_count,
        minhash_dup_count=minhash_dup_count,
        minhash_available=minhash_available,
        issue_counts=issue_counts,
        outlier_count=outlier_count,
        sem_dup_count=sem_dup_count,
        embeddings_available=embeddings_available,
        skip_embeddings=skip_embeddings,
    )

    result = DiagnoseResult(
        input_path=input_path,
        total_documents=total,
        elapsed_seconds=round(elapsed, 2),
        grade=grade,
        grade_color=grade_color,
        overall_score=round(overall_pct, 1),
        quality_distribution=quality_dist,
        issue_counts=issue_counts,
        worst_documents=worst,
        exact_duplicate_count=exact_dup_count,
        minhash_available=minhash_available,
        minhash_duplicate_count=minhash_dup_count,
        embeddings_available=embeddings_available,
        outlier_count=outlier_count,
        outlier_indices=outlier_idx,
        outlier_scores=outlier_sc,
        semantic_duplicate_count=sem_dup_count,
        semantic_duplicate_pairs=sem_dup_pairs,
        recommendations=recs,
    )

    if not quiet:
        _render_report(result, scored_docs, documents, text_field)

    return result


def _build_recommendations(
    *,
    input_path: str,
    total: int,
    exact_dup_count: int,
    minhash_dup_count: int | None,
    minhash_available: bool,
    issue_counts: dict[str, int],
    outlier_count: int | None,
    sem_dup_count: int | None,
    embeddings_available: bool,
    skip_embeddings: bool,
) -> list[str]:
    recs: list[str] = []

    if exact_dup_count > 0:
        recs.append(f"Remove {exact_dup_count:,} exact duplicates: dokime curate {input_path} clean.jsonl --dedup")

    if issue_counts.get("low_quality", 0) > 0:
        n = issue_counts["low_quality"]
        recs.append(f"Review {n:,} low-quality documents: dokime explore {input_path}")

    if minhash_dup_count and minhash_dup_count > 0:
        recs.append(
            f"Remove {minhash_dup_count:,} near-duplicates: dokime curate {input_path} clean.jsonl --fuzzy-dedup 0.8"
        )

    if sem_dup_count and sem_dup_count > 0:
        recs.append(f"Investigate {sem_dup_count:,} semantic duplicate pairs: dokime explore {input_path}")

    if outlier_count and outlier_count > 0:
        recs.append(f"Review {outlier_count:,} outlier documents: dokime outliers {input_path}")

    short = issue_counts.get("short_docs", 0)
    if short > total * 0.1:
        recs.append(f"Filter {short:,} very short docs: dokime curate {input_path} clean.jsonl --min-length 50")

    # Suggest installing missing deps
    if not minhash_available:
        recs.append("Install dokime[dedup] for fuzzy deduplication detection")
    if not embeddings_available and not skip_embeddings:
        recs.append("Install dokime[embeddings] for outlier detection and semantic dedup")

    return recs


def _render_report(
    result: DiagnoseResult,
    scored_docs: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    text_field: str,
) -> None:
    """Render the full diagnostic report to the terminal."""
    total = result.total_documents

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold {result.grade_color}]  {result.grade}  [/]\n[dim]({result.overall_score:.0f}/100)[/]",
            title="[bold]Dokime Health Check[/]",
            subtitle=f"{total:,} documents | {result.elapsed_seconds:.1f}s",
            width=60,
            padding=(1, 2),
        )
    )

    # === QUALITY ===
    console.print()
    console.print(Rule("[bold]1. Quality[/]", style="blue"))
    console.print()

    issues = result.issue_counts
    issues_table = Table(show_header=True, width=58, box=None)
    issues_table.add_column("Severity", style="bold", width=10)
    issues_table.add_column("Issue", width=28)
    issues_table.add_column("Count", justify="right", width=12)
    issues_table.add_column("", width=8)

    if result.exact_duplicate_count > 0:
        pct = result.exact_duplicate_count / total * 100
        color = "red" if pct > 5 else "yellow"
        sev = "CRITICAL" if pct > 5 else "WARNING"
        issues_table.add_row(
            f"[{color}]{sev}[/]",
            "Exact duplicates",
            f"{result.exact_duplicate_count:,} ({pct:.1f}%)",
            _bar(result.exact_duplicate_count, total, 8),
        )

    if issues.get("low_quality", 0) > 0:
        n = issues["low_quality"]
        pct = n / total * 100
        color = "red" if pct > 5 else "yellow"
        sev = "CRITICAL" if pct > 5 else "WARNING"
        issues_table.add_row(f"[{color}]{sev}[/]", "Low quality (< 0.3)", f"{n:,} ({pct:.1f}%)", _bar(n, total, 8))

    if issues.get("short_docs", 0) > 0:
        n = issues["short_docs"]
        pct = n / total * 100
        issues_table.add_row(
            "[yellow]WARNING[/]" if pct > 10 else "[blue]INFO[/]",
            "Very short (< 10 words)",
            f"{n:,} ({pct:.1f}%)",
            _bar(n, total, 8),
        )

    if issues.get("low_entropy", 0) > 0:
        n = issues["low_entropy"]
        issues_table.add_row(
            "[yellow]WARNING[/]", "Low entropy (repetitive)", f"{n:,} ({n / total * 100:.1f}%)", _bar(n, total, 8)
        )

    if issues.get("high_special", 0) > 0:
        n = issues["high_special"]
        issues_table.add_row(
            "[blue]INFO[/]", "High special chars", f"{n:,} ({n / total * 100:.1f}%)", _bar(n, total, 8)
        )

    # New Gopher/C4/FineWeb signal issues
    if issues.get("high_repetition", 0) > 0:
        n = issues["high_repetition"]
        pct = n / total * 100
        issues_table.add_row(
            "[red]CRITICAL[/]" if pct > 5 else "[yellow]WARNING[/]",
            "High repetition (Gopher)",
            f"{n:,} ({pct:.1f}%)",
            _bar(n, total, 8),
        )

    if issues.get("low_punctuation", 0) > 0:
        n = issues["low_punctuation"]
        pct = n / total * 100
        issues_table.add_row(
            "[yellow]WARNING[/]", "Low punctuation (FineWeb)", f"{n:,} ({pct:.1f}%)", _bar(n, total, 8)
        )

    if issues.get("high_dup_lines", 0) > 0:
        n = issues["high_dup_lines"]
        pct = n / total * 100
        issues_table.add_row(
            "[yellow]WARNING[/]", "Duplicate lines (Gopher)", f"{n:,} ({pct:.1f}%)", _bar(n, total, 8)
        )

    if issues.get("boilerplate", 0) > 0:
        n = issues["boilerplate"]
        issues_table.add_row(
            "[blue]INFO[/]", "Boilerplate text (C4)", f"{n:,} ({n / total * 100:.1f}%)", _bar(n, total, 8)
        )

    if issues.get("few_sentences", 0) > 0:
        n = issues["few_sentences"]
        issues_table.add_row(
            "[blue]INFO[/]", "Few sentences (C4)", f"{n:,} ({n / total * 100:.1f}%)", _bar(n, total, 8)
        )

    has_issues = any(v > 0 for v in issues.values()) or result.exact_duplicate_count > 0
    if not has_issues:
        issues_table.add_row("[green]NONE[/]", "No issues found", "-", "")

    console.print(issues_table)

    # Quality distribution
    console.print()
    dist = result.quality_distribution
    dist_table = Table(show_header=False, width=58, box=None)
    dist_table.add_column("Range", width=20)
    dist_table.add_column("Count", justify="right", width=8)
    dist_table.add_column("", width=25)
    dist_table.add_row("[green]Excellent (>0.8)[/]", f"{dist['excellent']:,}", _bar(dist["excellent"], total, 25))
    dist_table.add_row("[blue]Good (0.6-0.8)[/]", f"{dist['good']:,}", _bar(dist["good"], total, 25))
    dist_table.add_row("[yellow]Medium (0.3-0.6)[/]", f"{dist['medium']:,}", _bar(dist["medium"], total, 25))
    dist_table.add_row("[red]Poor (<0.3)[/]", f"{dist['poor']:,}", _bar(dist["poor"], total, 25))
    console.print(dist_table)

    # Worst documents
    if result.worst_documents:
        console.print()
        worst_table = Table(title=f"Worst {len(result.worst_documents)}", show_header=True, width=58, box=None)
        worst_table.add_column("#", width=3)
        worst_table.add_column("Score", width=6)
        worst_table.add_column("Preview", width=45)
        for rank, (idx, sc) in enumerate(result.worst_documents, 1):
            text = scored_docs[idx].get(text_field, "")[:80].replace("\n", " ")
            worst_table.add_row(str(rank), f"[red]{sc:.2f}[/]", text)
        console.print(worst_table)

    # === DUPLICATES ===
    console.print()
    console.print(Rule("[bold]2. Duplicates[/]", style="blue"))
    console.print()

    console.print(
        f"  Exact duplicates:    [bold]{result.exact_duplicate_count:,}[/] ({result.exact_duplicate_count / total * 100:.1f}%)"
    )

    if result.minhash_available:
        n = result.minhash_duplicate_count or 0
        console.print(f"  MinHash near-dupes:  [bold]{n:,}[/] ({n / total * 100:.1f}%)")
    else:
        console.print("  MinHash near-dupes:  [yellow]skipped[/] (pip install dokime[dedup])")

    if result.embeddings_available and result.semantic_duplicate_count is not None:
        console.print(f"  Semantic near-dupes: [bold]{result.semantic_duplicate_count:,}[/] pairs")
    elif not result.embeddings_available:
        console.print("  Semantic near-dupes: [yellow]skipped[/] (pip install dokime[embeddings])")

    unique = total - result.exact_duplicate_count
    console.print(f"\n  Unique documents:    [bold green]{unique:,}[/] / {total:,}")

    # === OUTLIERS ===
    console.print()
    console.print(Rule("[bold]3. Outliers[/]", style="blue"))
    console.print()

    if result.embeddings_available and result.outlier_count is not None:
        console.print(f"  Found [bold]{result.outlier_count:,}[/] outliers (>2 sigma from mean)\n")

        if result.outlier_indices:
            out_table = Table(show_header=True, width=58, box=None)
            out_table.add_column("#", width=3)
            out_table.add_column("Score", width=7)
            out_table.add_column("Preview", width=44)
            for rank, (idx, sc) in enumerate(zip(result.outlier_indices, result.outlier_scores, strict=True), 1):
                text = documents[idx].get(text_field, "")[:80].replace("\n", " ")
                out_table.add_row(str(rank), f"[red]{sc:.4f}[/]", text)
            console.print(out_table)
    else:
        console.print("  [yellow]Skipped[/] — requires dokime[embeddings]")

    # === RECOMMENDATIONS ===
    console.print()
    console.print(Rule("[bold]4. Recommendations[/]", style="blue"))
    console.print()

    if result.recommendations:
        for i, rec in enumerate(result.recommendations, 1):
            console.print(f"  {i}. {rec}")
    else:
        console.print("  [green]No issues found. Your dataset looks healthy.[/]")

    console.print()
