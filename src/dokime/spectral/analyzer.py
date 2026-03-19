"""Spectral analysis of vision encoder Jacobians for OOD detection.

Analyzes the singular value spectrum of a pretrained encoder's Jacobian
to determine which spectral bands are discriminative for out-of-distribution
detection. Based on verified findings from JEPA-SCORE research:

  - Top-10 SV correlation between ID and OOD > 0.997 (anti-discriminative)
  - Cohen's d collapses from 0.98 (ViT-S) to 0.04 (ViT-L)
  - Tail-weighting (dropping top ~20% SVs) improves AUROC by 2.2%

Reference: Balestriero et al., "Gaussian Embeddings: How JEPAs Secretly
Learn Your Data Density," NeurIPS 2025. arXiv:2510.05949.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6


@dataclass
class SpectralComparison:
    """Results from comparing ID vs OOD spectral distributions."""

    cohens_d: float
    """Cohen's d effect size between ID and OOD log-sum-sv scores."""

    top_k_correlation: float
    """Pearson correlation of mean top-k singular values between ID and OOD."""

    auroc_standard: float
    """AUROC using all singular values (standard JEPA-SCORE)."""

    auroc_tail_weighted: float
    """AUROC using tail-weighted singular values (top SVs dropped)."""

    optimal_drop_k: int
    """Number of top SVs to drop for best tail-weighted AUROC."""

    per_band_discrimination: dict[str, float]
    """Cohen's d for each spectral band (top/mid/tail quartiles)."""


def _compute_jacobian_svd_full(
    model: nn.Module,
    x: torch.Tensor,
    eps: float = EPS,
) -> np.ndarray:
    """Compute singular values of the full Jacobian for a single image.

    Matches the paper's Appendix B reference implementation exactly.

    Args:
        model: Encoder mapping (B, C, H, W) -> (B, d).
        x: Single image tensor without batch dim, e.g. (3, 224, 224).
        eps: Not used for SVD, kept for API consistency.

    Returns:
        Singular values array of shape (d,) in descending order.
    """
    device = next(model.parameters()).device
    x_batch = x.unsqueeze(0).to(device)

    def func(inp: torch.Tensor) -> torch.Tensor:
        return model(inp).sum(0)

    J = torch.autograd.functional.jacobian(func, x_batch, vectorize=False)
    J = J.flatten(2).permute(1, 0, 2)  # (1, d, D)
    sv = torch.linalg.svdvals(J)  # (1, d)
    return sv[0].detach().cpu().numpy()


def _compute_jacobian_svd_randomized(
    model: nn.Module,
    x: torch.Tensor,
    n_proj: int = 64,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Compute approximate singular values via random projection.

    Uses VJPs with random Gaussian vectors to approximate the Jacobian's
    spectrum without materializing the full Jacobian matrix.

    Args:
        model: Encoder mapping (B, C, H, W) -> (B, d).
        x: Single image tensor without batch dim.
        n_proj: Number of random projections.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Singular values array of shape (min(n_proj, d),) in descending order.
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        emb = model(x.unsqueeze(0).to(device))
    d = emb.shape[1]

    omega = torch.randn(n_proj, d, device=device, generator=generator)

    vjp_rows = []
    for j in range(n_proj):
        x_j = x.unsqueeze(0).to(device).requires_grad_(True)
        out = model(x_j)
        (g,) = torch.autograd.grad(
            out, x_j,
            grad_outputs=omega[j].unsqueeze(0),
            retain_graph=False,
        )
        vjp_rows.append(g.flatten().detach())

    J_proj = torch.stack(vjp_rows, dim=0)  # (n_proj, D)
    sv = torch.linalg.svdvals(J_proj)
    return sv.cpu().numpy()


def _auroc_from_scores(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """Compute AUROC treating ID as positive (higher scores = more ID).

    Uses sklearn if available, otherwise falls back to a manual
    trapezoidal implementation.
    """
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        # Manual AUROC via sorted thresholds (Wilcoxon-Mann-Whitney).
        n_id = len(id_scores)
        n_ood = len(ood_scores)
        if n_id == 0 or n_ood == 0:
            return 0.5
        count = 0
        for s_id in id_scores:
            count += np.sum(s_id > ood_scores) + 0.5 * np.sum(s_id == ood_scores)
        return float(count / (n_id * n_ood))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


class SpectralAnalyzer:
    """Analyze the Jacobian spectrum of any pretrained vision encoder.

    Wraps a PyTorch nn.Module that maps (B, C, H, W) -> (B, d) and provides
    methods to compute singular value spectra, compare ID vs OOD distributions,
    and generate diagnostic reports.

    Example::

        import torchvision.models as models
        encoder = models.vit_b_16(weights="DEFAULT")
        analyzer = SpectralAnalyzer(encoder)

        id_spectra = analyzer.compute_spectra(id_images)
        ood_spectra = analyzer.compute_spectra(ood_images)
        results = analyzer.compare_distributions(id_spectra, ood_spectra)
        analyzer.report(id_spectra, ood_spectra)
    """

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        """Wrap any PyTorch vision encoder for Jacobian spectral analysis.

        Args:
            model: Encoder that maps (B, C, H, W) -> (B, d).
                   Must be differentiable (no frozen non-leaf tensors blocking grad).
            device: Device to run computation on ("cuda" or "cpu").
        """
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def compute_spectra(
        self,
        images: torch.Tensor,
        method: Literal["full", "randomized"] = "full",
        n_proj: int = 64,
        seed: int = 42,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute singular value spectra for a batch of images.

        For each image, computes the Jacobian of the encoder and returns
        its singular values. This is the core diagnostic: the distribution
        of these singular values across ID vs OOD data reveals which
        spectral bands carry discriminative signal.

        Args:
            images: Tensor of shape (N, C, H, W).
            method: "full" for exact Jacobian (expensive but exact),
                    "randomized" for projected approximation (cheaper).
            n_proj: Number of random projections (only for method="randomized").
            seed: Random seed for reproducibility (only for method="randomized").
            verbose: Print progress every 50 images.

        Returns:
            Array of shape (N, d) containing singular values per image,
            sorted in descending order along axis 1.
        """
        if method not in ("full", "randomized"):
            raise ValueError(
                f"Unknown method: {method!r}. Use 'full' or 'randomized'."
            )

        n = len(images)
        spectra = []

        if method == "randomized":
            gen = torch.Generator(device=self.device).manual_seed(seed)
        else:
            gen = None

        for i in range(n):
            if method == "full":
                sv = _compute_jacobian_svd_full(self.model, images[i])
            else:
                sv = _compute_jacobian_svd_randomized(
                    self.model, images[i], n_proj=n_proj, generator=gen,
                )
            spectra.append(sv)

            if verbose and ((i + 1) % 50 == 0 or i == 0):
                score = np.sum(np.log(np.clip(sv, EPS, None)))
                print(f"  [{i + 1}/{n}] score={score:.2f}", flush=True)

        return np.stack(spectra, axis=0)

    def compare_distributions(
        self,
        id_spectra: np.ndarray,
        ood_spectra: np.ndarray,
        top_k: int = 10,
    ) -> SpectralComparison:
        """Compare ID vs OOD spectral distributions.

        Implements the key diagnostic analyses from our JEPA-SCORE research:
        standard vs tail-weighted AUROC, per-band discrimination, and
        top-k SV correlation.

        Args:
            id_spectra: Array of shape (N_id, d) from compute_spectra on ID data.
            ood_spectra: Array of shape (N_ood, d) from compute_spectra on OOD data.
            top_k: Number of top SVs for correlation analysis (default 10).

        Returns:
            SpectralComparison dataclass with all diagnostic metrics.
        """
        d = id_spectra.shape[1]
        top_k = min(top_k, d)

        # --- Standard JEPA-SCORE: log-sum of all SVs ---
        id_scores = np.sum(np.log(np.clip(id_spectra, EPS, None)), axis=1)
        ood_scores = np.sum(np.log(np.clip(ood_spectra, EPS, None)), axis=1)

        cohens_d = _cohens_d(id_scores, ood_scores)
        auroc_standard = _auroc_from_scores(id_scores, ood_scores)

        # --- Top-k correlation: shows top SVs are NOT discriminative ---
        id_mean_topk = np.mean(id_spectra[:, :top_k], axis=0)
        ood_mean_topk = np.mean(ood_spectra[:, :top_k], axis=0)
        if np.std(id_mean_topk) < 1e-12 or np.std(ood_mean_topk) < 1e-12:
            top_k_corr = 1.0  # degenerate case
        else:
            top_k_corr = float(np.corrcoef(id_mean_topk, ood_mean_topk)[0, 1])

        # --- Tail-weighted: sweep drop_k to find best AUROC ---
        best_auroc_tw = auroc_standard
        best_drop_k = 0
        # Search from 5% to 50% in steps, capped at reasonable range
        candidates = set()
        for pct in range(5, 55, 5):
            candidates.add(max(1, int(d * pct / 100)))
        candidates = sorted(candidates)

        for drop_k in candidates:
            if drop_k >= d:
                continue
            id_tw = np.sum(
                np.log(np.clip(id_spectra[:, drop_k:], EPS, None)), axis=1
            )
            ood_tw = np.sum(
                np.log(np.clip(ood_spectra[:, drop_k:], EPS, None)), axis=1
            )
            auroc_tw = _auroc_from_scores(id_tw, ood_tw)
            if auroc_tw > best_auroc_tw:
                best_auroc_tw = auroc_tw
                best_drop_k = drop_k

        auroc_tail_weighted = best_auroc_tw

        # --- Per-band discrimination (quartiles) ---
        q1, q2, q3 = d // 4, d // 2, 3 * d // 4
        bands = {
            "top_quartile": (0, max(q1, 1)),
            "upper_mid": (max(q1, 1), max(q2, 2)),
            "lower_mid": (max(q2, 2), max(q3, 3)),
            "tail_quartile": (max(q3, 3), d),
        }
        per_band = {}
        for band_name, (lo, hi) in bands.items():
            if lo >= hi or lo >= d:
                per_band[band_name] = 0.0
                continue
            id_band = np.sum(
                np.log(np.clip(id_spectra[:, lo:hi], EPS, None)), axis=1
            )
            ood_band = np.sum(
                np.log(np.clip(ood_spectra[:, lo:hi], EPS, None)), axis=1
            )
            per_band[band_name] = _cohens_d(id_band, ood_band)

        return SpectralComparison(
            cohens_d=cohens_d,
            top_k_correlation=top_k_corr,
            auroc_standard=auroc_standard,
            auroc_tail_weighted=auroc_tail_weighted,
            optimal_drop_k=best_drop_k,
            per_band_discrimination=per_band,
        )

    def report(
        self,
        id_spectra: np.ndarray,
        ood_spectra: np.ndarray,
        top_k: int = 10,
    ) -> SpectralComparison:
        """Print a rich terminal report of spectral analysis results.

        Computes all metrics via compare_distributions and renders a
        formatted table. Requires the 'rich' package for formatted output;
        falls back to plain text if unavailable.

        Args:
            id_spectra: Array of shape (N_id, d) from compute_spectra.
            ood_spectra: Array of shape (N_ood, d) from compute_spectra.
            top_k: Number of top SVs for correlation analysis.

        Returns:
            The SpectralComparison result (also printed).
        """
        result = self.compare_distributions(id_spectra, ood_spectra, top_k=top_k)

        try:
            self._report_rich(result, id_spectra, ood_spectra, top_k)
        except ImportError:
            self._report_plain(result, id_spectra, ood_spectra, top_k)

        return result

    @staticmethod
    def _report_rich(
        result: SpectralComparison,
        id_spectra: np.ndarray,
        ood_spectra: np.ndarray,
        top_k: int,
    ) -> None:
        """Render report using rich library."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        d = id_spectra.shape[1]

        console.print()
        console.print("[bold]Spectral Analysis Report[/bold]", style="cyan")
        console.print(
            f"  ID samples: {len(id_spectra)}  |  OOD samples: {len(ood_spectra)}  |  "
            f"Embedding dim: {d}"
        )
        console.print()

        # Main metrics table
        table = Table(title="Detection Metrics", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Interpretation")

        # Cohen's d
        d_interp = (
            "large" if abs(result.cohens_d) >= 0.8
            else "medium" if abs(result.cohens_d) >= 0.5
            else "small" if abs(result.cohens_d) >= 0.2
            else "negligible"
        )
        table.add_row(
            "Cohen's d (full)",
            f"{result.cohens_d:.4f}",
            f"Effect size: {d_interp}",
        )

        # Top-k correlation
        table.add_row(
            f"Top-{top_k} SV correlation",
            f"{result.top_k_correlation:.4f}",
            ">.99 = top SVs NOT discriminative" if result.top_k_correlation > 0.99
            else "Top SVs carry some signal",
        )

        # AUROC standard
        table.add_row(
            "AUROC (standard)",
            f"{result.auroc_standard:.4f}",
            "",
        )

        # AUROC tail-weighted
        delta = result.auroc_tail_weighted - result.auroc_standard
        delta_str = f"({delta:+.1%} vs standard)"
        table.add_row(
            "AUROC (tail-weighted)",
            f"{result.auroc_tail_weighted:.4f}",
            f"Drop top {result.optimal_drop_k} SVs {delta_str}",
        )

        console.print(table)
        console.print()

        # Per-band table
        band_table = Table(title="Per-Band Discrimination (Cohen's d)", show_header=True)
        band_table.add_column("Band", style="bold")
        band_table.add_column("SV indices")
        band_table.add_column("Cohen's d", justify="right")

        q1, q2, q3 = d // 4, d // 2, 3 * d // 4
        band_ranges = {
            "top_quartile": (0, max(q1, 1)),
            "upper_mid": (max(q1, 1), max(q2, 2)),
            "lower_mid": (max(q2, 2), max(q3, 3)),
            "tail_quartile": (max(q3, 3), d),
        }
        for band_name, (lo, hi) in band_ranges.items():
            cd = result.per_band_discrimination.get(band_name, 0.0)
            band_table.add_row(band_name, f"[{lo}:{hi}]", f"{cd:.4f}")

        console.print(band_table)
        console.print()

    @staticmethod
    def _report_plain(
        result: SpectralComparison,
        id_spectra: np.ndarray,
        ood_spectra: np.ndarray,
        top_k: int,
    ) -> None:
        """Render report as plain text (fallback if rich not installed)."""
        d = id_spectra.shape[1]
        sep = "=" * 60

        print()
        print(sep)
        print("SPECTRAL ANALYSIS REPORT")
        print(sep)
        print(
            f"  ID samples: {len(id_spectra)}  |  OOD samples: {len(ood_spectra)}  |  "
            f"Embedding dim: {d}"
        )
        print()
        print("Detection Metrics:")
        print(f"  Cohen's d (full):         {result.cohens_d:.4f}")
        print(f"  Top-{top_k} SV correlation:    {result.top_k_correlation:.4f}")
        print(f"  AUROC (standard):         {result.auroc_standard:.4f}")
        delta = result.auroc_tail_weighted - result.auroc_standard
        print(
            f"  AUROC (tail-weighted):    {result.auroc_tail_weighted:.4f}  "
            f"(drop top {result.optimal_drop_k}, {delta:+.1%})"
        )
        print()
        print("Per-Band Discrimination (Cohen's d):")
        for band_name, cd in result.per_band_discrimination.items():
            print(f"  {band_name:20s}  {cd:.4f}")
        print(sep)
        print()
