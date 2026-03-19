"""Data quality report — the sharp hook of Dokime.

`dokime score data.jsonl` produces a beautiful terminal report showing:
- Overall grade (A-F)
- Issue breakdown (duplicates, low quality, outliers, short docs)
- Quality distribution histogram
- Worst documents preview
- Actionable recommendations
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dokime.quality.scoring import QualityScorer

console = Console()


def _grade_letter(score: float) -> tuple[str, str]:
    """Convert a 0-100 score to a letter grade and color."""
    if score >= 90:
        return "A", "green"
    if score >= 80:
        return "B+", "green"
    if score >= 70:
        return "B", "blue"
    if score >= 60:
        return "B-", "blue"
    if score >= 50:
        return "C+", "yellow"
    if score >= 40:
        return "C", "yellow"
    if score >= 30:
        return "D", "red"
    return "F", "red"


def _bar(count: int, total: int, width: int = 20) -> str:
    """Create a simple bar chart string."""
    if total == 0:
        return " " * width
    filled = int((count / total) * width)
    return "#" * filled + "-" * (width - filled)


def run_report(
    input_path: str,
    text_field: str = "text",
    show_worst: int = 5,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run a full data quality report on a dataset.

    This is the core of `dokime score` — the one-command quality assessment.

    Returns:
        Dict with all computed statistics.
    """
    from dokime.io.readers import auto_read

    start = time.time()
    scorer = QualityScorer(text_field=text_field)

    # Collect all scored documents
    documents: list[dict[str, Any]] = []
    scores: list[float] = []
    hashes: set[str] = set()
    duplicate_count = 0

    for doc in auto_read(input_path):
        scored = scorer.score(doc)
        documents.append(scored)
        scores.append(scored["_quality_score"])

        # Check for exact duplicates
        text = doc.get(text_field, "")
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if h in hashes:
            duplicate_count += 1
        else:
            hashes.add(h)

    elapsed = time.time() - start
    total = len(documents)

    if total == 0:
        if not quiet:
            console.print("[yellow]Empty dataset. Nothing to score.[/]")
        return {"total": 0, "grade": "N/A"}

    # Compute statistics
    avg_score = sum(scores) / total
    overall_pct = avg_score * 100

    # Count issues
    short_docs = sum(1 for d in documents if d["_word_count"] < 10)
    low_quality = sum(1 for s in scores if s < 0.3)
    medium_quality = sum(1 for s in scores if 0.3 <= s < 0.6)
    good_quality = sum(1 for s in scores if 0.6 <= s < 0.8)
    excellent_quality = sum(1 for s in scores if s >= 0.8)
    low_entropy = sum(1 for d in documents if d.get("_char_entropy", 5) < 2.0)
    high_special = sum(1 for d in documents if d.get("_special_ratio", 0) > 0.3)

    grade, grade_color = _grade_letter(overall_pct)

    # Find worst documents
    indexed_scores = [(i, scores[i]) for i in range(total)]
    indexed_scores.sort(key=lambda x: x[1])
    worst = indexed_scores[:show_worst]

    # Build result dict
    result: dict[str, Any] = {
        "total": total,
        "elapsed_seconds": round(elapsed, 2),
        "grade": grade,
        "overall_score": round(overall_pct, 1),
        "duplicates": duplicate_count,
        "short_docs": short_docs,
        "low_quality": low_quality,
        "low_entropy": low_entropy,
        "high_special_chars": high_special,
        "distribution": {
            "excellent": excellent_quality,
            "good": good_quality,
            "medium": medium_quality,
            "poor": low_quality,
        },
    }

    if quiet:
        return result

    # ===== RENDER THE REPORT =====

    console.print()
    console.print(
        Panel(
            f"[bold {grade_color}]  {grade}  [/]\n[dim]({overall_pct:.0f}/100)[/]",
            title="[bold]Dokime Data Quality Report[/]",
            subtitle=f"{total:,} documents | {elapsed:.1f}s",
            width=60,
            padding=(1, 2),
        )
    )

    console.print()
    console.print(f"  [bold]Dataset:[/] {input_path}")
    console.print(f"  [bold]Documents:[/] {total:,}")
    console.print(f"  [bold]Scanned in:[/] {elapsed:.1f}s")
    console.print()

    # Issues table
    issues_table = Table(title="Issues Found", show_header=True, width=58)
    issues_table.add_column("Severity", style="bold", width=10)
    issues_table.add_column("Issue", width=28)
    issues_table.add_column("Count", justify="right", width=8)
    issues_table.add_column("", width=10)

    if duplicate_count > 0:
        pct = duplicate_count / total * 100
        severity = "CRITICAL" if pct > 5 else "WARNING"
        color = "red" if pct > 5 else "yellow"
        issues_table.add_row(
            f"[{color}]{severity}[/]",
            "Exact duplicates",
            f"{duplicate_count:,} ({pct:.1f}%)",
            _bar(duplicate_count, total, 10),
        )

    if low_quality > 0:
        pct = low_quality / total * 100
        severity = "CRITICAL" if pct > 5 else "WARNING"
        color = "red" if pct > 5 else "yellow"
        issues_table.add_row(
            f"[{color}]{severity}[/]",
            "Low quality (score < 0.3)",
            f"{low_quality:,} ({pct:.1f}%)",
            _bar(low_quality, total, 10),
        )

    if short_docs > 0:
        pct = short_docs / total * 100
        issues_table.add_row(
            "[yellow]WARNING[/]" if pct > 10 else "[blue]INFO[/]",
            "Very short (<10 words)",
            f"{short_docs:,} ({pct:.1f}%)",
            _bar(short_docs, total, 10),
        )

    if low_entropy > 0:
        pct = low_entropy / total * 100
        issues_table.add_row(
            "[yellow]WARNING[/]",
            "Low entropy (repetitive)",
            f"{low_entropy:,} ({pct:.1f}%)",
            _bar(low_entropy, total, 10),
        )

    if high_special > 0:
        pct = high_special / total * 100
        issues_table.add_row(
            "[blue]INFO[/]",
            "High special chars (>30%)",
            f"{high_special:,} ({pct:.1f}%)",
            _bar(high_special, total, 10),
        )

    if duplicate_count == 0 and low_quality == 0 and short_docs == 0:
        issues_table.add_row("[green]NONE[/]", "No issues found", "-", "")

    console.print(issues_table)
    console.print()

    # Quality distribution
    dist_table = Table(title="Quality Distribution", show_header=True, width=58)
    dist_table.add_column("Range", width=18)
    dist_table.add_column("Count", justify="right", width=8)
    dist_table.add_column("", width=25)

    dist_table.add_row(
        "[green]Excellent (>0.8)[/]",
        f"{excellent_quality:,}",
        _bar(excellent_quality, total, 25),
    )
    dist_table.add_row(
        "[blue]Good (0.6-0.8)[/]",
        f"{good_quality:,}",
        _bar(good_quality, total, 25),
    )
    dist_table.add_row(
        "[yellow]Medium (0.3-0.6)[/]",
        f"{medium_quality:,}",
        _bar(medium_quality, total, 25),
    )
    dist_table.add_row(
        "[red]Poor (<0.3)[/]",
        f"{low_quality:,}",
        _bar(low_quality, total, 25),
    )

    console.print(dist_table)
    console.print()

    # Worst documents
    if worst:
        worst_table = Table(title=f"Worst {show_worst} Documents", show_header=True, width=58)
        worst_table.add_column("#", width=4)
        worst_table.add_column("Score", width=6)
        worst_table.add_column("Preview", width=44)

        for rank, (idx, score) in enumerate(worst, 1):
            text = documents[idx].get(text_field, "")[:80].replace("\n", " ")
            worst_table.add_row(
                str(rank),
                f"[red]{score:.2f}[/]",
                text,
            )

        console.print(worst_table)
        console.print()

    # Recommendations
    recs = []
    if duplicate_count > 0:
        recs.append(
            f"Remove {duplicate_count:,} duplicates -> [bold]dokime curate {input_path} clean.jsonl --dedup[/]"
        )
    if low_quality > 0:
        recs.append(f"Review {low_quality:,} low-quality docs -> [bold]dokime explore {input_path}[/]")
    if short_docs > total * 0.1:
        recs.append(
            f"Filter {short_docs:,} short docs -> [bold]dokime curate {input_path} clean.jsonl --min-length 50[/]"
        )

    if recs:
        console.print("[bold]Recommendations:[/]")
        for i, rec in enumerate(recs, 1):
            console.print(f"  {i}. {rec}")
        console.print()

    return result
