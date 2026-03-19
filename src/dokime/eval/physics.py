"""PhyX physics benchmark evaluation for vision-language models.

Wraps the PhyX MC (multiple-choice) benchmark:
  - 1000 questions across 6 physics domains
  - Images sent as base64 to an OpenAI-compatible vision API
  - Rule-based STR answer extraction (no LLM judge needed)

Reference: https://github.com/PhyX-Bench/PhyX
"""

from __future__ import annotations

import csv
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Default dataset path (PhyX clone location)
# ---------------------------------------------------------------------------
_DEFAULT_DATASET = Path(__file__).resolve().parents[4] / "PhyX" / "dataset" / "PhyX_MC.tsv"
# Fallback: common clone location on this machine
_FALLBACK_DATASET = Path.home() / "PhyX" / "dataset" / "PhyX_MC.tsv"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    index: int
    category: str
    subfield: str
    ground_truth: str
    prediction: str
    extracted: str
    correct: bool


@dataclass
class DomainScore:
    category: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    domains: dict[str, DomainScore] = field(default_factory=dict)
    results: list[QuestionResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        domain_data = {}
        for name, ds in sorted(self.domains.items()):
            domain_data[name] = {
                "correct": ds.correct,
                "total": ds.total,
                "accuracy": round(ds.accuracy * 100, 2),
            }
        return {
            "overall": {
                "correct": self.correct,
                "total": self.total,
                "accuracy": round(self.accuracy * 100, 2),
            },
            "domains": domain_data,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load PhyX MC TSV dataset, returning list of row dicts.

    Columns: index, image (base64 JPEG), question, answer, category, subfield, reasoning_type
    """
    if path is None:
        if _DEFAULT_DATASET.exists():
            path = _DEFAULT_DATASET
        elif _FALLBACK_DATASET.exists():
            path = _FALLBACK_DATASET
        else:
            raise FileNotFoundError(
                "PhyX dataset not found. Provide --dataset path or clone PhyX to ~/PhyX"
            )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # The base64 image fields can be very large
    csv.field_size_limit(sys.maxsize)

    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Answer extraction (ported from PhyX PhyX_process_line_MC)
# ---------------------------------------------------------------------------

def extract_mc_answer(prediction: str) -> str:
    """Extract a single letter A/B/C/D from a model's free-form response.

    Uses the same heuristics as PhyX's STR method:
    1. Look for patterns like "correct answer is X", "option X", etc.
    2. Fall back to "X:" patterns
    3. Fall back to direct single-letter match
    """
    pred = prediction.strip()

    # Strategy 1: keyword + letter pattern (PhyX's primary regex)
    pattern = r'\b(?:correct|answer|option|Answer|Option|Correct)\b[\s\S]*?([A-D])'
    match = re.search(pattern, pred)
    if match:
        return match.group(1).upper()

    # Strategy 2: "A:" style patterns (PhyX fallback)
    matches = re.findall(r'([ABCD]):', pred)
    if matches:
        return matches[-1].upper()

    # Strategy 3: bold markers like "A**" or "**A"
    bold_match = re.search(r'\*\*([A-D])\b|\b([A-D])\*\*', pred)
    if bold_match:
        letter = bold_match.group(1) or bold_match.group(2)
        return letter.upper()

    # Strategy 4: if the entire response is just a single letter
    if pred.upper() in {"A", "B", "C", "D"}:
        return pred.upper()

    # Strategy 5: first occurrence of a standalone A-D letter
    standalone = re.search(r'\b([A-D])\b', pred)
    if standalone:
        return standalone.group(1).upper()

    return pred  # Return raw if we can't extract


def score_mc(ground_truth: str, prediction: str) -> tuple[str, bool]:
    """Score a multiple-choice prediction against ground truth.

    Returns (extracted_answer, is_correct).
    """
    extracted = extract_mc_answer(prediction)
    gt = ground_truth.strip().upper()

    # Direct match
    if gt == extracted.strip().upper():
        return extracted, True

    # Check if ground truth letter appears in specific patterns in raw prediction
    raw = prediction.strip()
    if f"{gt}:" in raw or f"{gt}**" in raw or f"**{gt}" in raw:
        return extracted, True

    return extracted, False


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def query_model(
    model_url: str,
    api_key: str | None,
    image_b64: str,
    question: str,
    timeout: int = 120,
    max_retries: int = 3,
) -> str:
    """Send image+question to an OpenAI-compatible vision API and return the text response."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build the message with image + text
    payload = {
        "model": "default",  # Most OpenAI-compatible servers ignore or auto-select
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": question + "\n\nAnswer with the letter of the correct option (A, B, C, or D).",
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                model_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Standard OpenAI format
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff

    return f"ERROR: {last_err}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    model_url: str,
    api_key: str | None = None,
    dataset_path: str | Path | None = None,
    limit: int = 0,
    on_progress: Any = None,
) -> EvalResult:
    """Run the PhyX MC evaluation against a VLM endpoint.

    Args:
        model_url: OpenAI-compatible chat completions endpoint
        api_key: Optional bearer token
        dataset_path: Path to PhyX_MC.tsv (uses bundled default if None)
        limit: Max questions to evaluate (0 = all)
        on_progress: Optional callable(current, total, result) for progress updates

    Returns:
        EvalResult with per-domain and overall scores
    """
    rows = load_dataset(dataset_path)
    if limit > 0:
        rows = rows[:limit]

    result = EvalResult()
    t0 = time.time()

    for i, row in enumerate(rows):
        idx = int(row["index"])
        category = row["category"]
        subfield = row["subfield"]
        gt_answer = row["answer"].strip()
        question = row["question"]
        image_b64 = row["image"]

        # Query the model
        prediction = query_model(model_url, api_key, image_b64, question)

        # Score
        extracted, correct = score_mc(gt_answer, prediction)

        qr = QuestionResult(
            index=idx,
            category=category,
            subfield=subfield,
            ground_truth=gt_answer,
            prediction=prediction,
            extracted=extracted,
            correct=correct,
        )
        result.results.append(qr)
        result.total += 1
        if correct:
            result.correct += 1

        # Track per-domain
        if category not in result.domains:
            result.domains[category] = DomainScore(category=category)
        ds = result.domains[category]
        ds.total += 1
        if correct:
            ds.correct += 1

        if on_progress:
            on_progress(i + 1, len(rows), qr)

    result.elapsed_seconds = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Rich terminal report
# ---------------------------------------------------------------------------

def print_report(result: EvalResult) -> None:
    """Print a rich terminal report of evaluation results."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold blue]Dokime[/] eval-physics  |  "
            f"[bold]{result.total}[/] questions  |  "
            f"[bold green]{result.accuracy:.1%}[/] overall accuracy  |  "
            f"{result.elapsed_seconds:.0f}s elapsed",
            title="PhyX Benchmark Results",
            border_style="blue",
        )
    )

    # Domain breakdown table
    table = Table(title="Accuracy by Physics Domain", show_header=True, header_style="bold")
    table.add_column("Domain", style="bold", min_width=20)
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")

    for name in sorted(result.domains):
        ds = result.domains[name]
        pct = f"{ds.accuracy:.1%}"
        # Color code: green >= 50%, yellow >= 25%, red < 25%
        if ds.accuracy >= 0.5:
            style = "green"
        elif ds.accuracy >= 0.25:
            style = "yellow"
        else:
            style = "red"
        table.add_row(name, str(ds.correct), str(ds.total), f"[{style}]{pct}[/]")

    # Overall row
    table.add_section()
    table.add_row(
        "[bold]Overall[/]",
        f"[bold]{result.correct}[/]",
        f"[bold]{result.total}[/]",
        f"[bold green]{result.accuracy:.1%}[/]",
    )

    console.print(table)
    console.print()

    # Show some wrong answers as examples
    wrong = [r for r in result.results if not r.correct]
    if wrong:
        n_show = min(5, len(wrong))
        console.print(f"[dim]Showing {n_show} incorrect answers (of {len(wrong)} total):[/]")
        for qr in wrong[:n_show]:
            console.print(
                f"  [red]X[/] Q{qr.index} [{qr.category}/{qr.subfield}] "
                f"GT={qr.ground_truth} Extracted={qr.extracted}"
            )
        console.print()
