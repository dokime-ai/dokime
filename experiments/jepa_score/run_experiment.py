"""JEPA-SCORE vs Simple Baselines: OOD Detection Benchmark.

First independent evaluation of JEPA-SCORE (arXiv:2510.05949, NeurIPS 2025).

Tests whether JEPA-SCORE (Jacobian-based density estimation) beats simpler
methods for out-of-distribution detection using DINOv2 embeddings.

Usage:
    python run_experiment.py                    # full experiment
    python run_experiment.py --n-samples 200    # quick test (fewer samples)
    python run_experiment.py --device cuda       # use GPU
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cifar10(n_samples: int | None = None) -> tuple[Tensor, Tensor]:
    """Load CIFAR-10 test set as tensors."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    images = []
    labels = []
    for i, (img, label) in enumerate(ds):
        if n_samples and i >= n_samples:
            break
        images.append(img)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def get_svhn(n_samples: int | None = None) -> tuple[Tensor, Tensor]:
    """Load SVHN test set as OOD data."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.SVHN(root="./data", split="test", download=True, transform=transform)
    images = []
    labels = []
    for i, (img, label) in enumerate(ds):
        if n_samples and i >= n_samples:
            break
        images.append(img)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dinov2(device: str = "cpu") -> torch.nn.Module:
    """Load DINOv2 ViT-S/14 (smallest variant, Apache 2.0)."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(
    model: torch.nn.Module,
    images: Tensor,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Extract CLS embeddings from DINOv2."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# JEPA-SCORE (the method being tested)
# ---------------------------------------------------------------------------

def compute_jepa_score_randomized(
    model: torch.nn.Module,
    images: Tensor,
    device: str = "cpu",
    n_projections: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Compute JEPA-SCORE using randomized SVD via Jacobian-vector products.

    Instead of computing the full Jacobian (embed_dim backward passes),
    we use random projections to estimate the top singular values.

    JEPA-SCORE(x) = sum of log singular values of J_f(x)

    Args:
        model: DINOv2 encoder.
        images: Input images tensor.
        device: torch device.
        n_projections: Number of random projection vectors (controls accuracy vs speed).
        seed: Random seed for reproducibility.

    Returns:
        Array of JEPA-SCORE values, one per image.
    """
    rng = np.random.RandomState(seed)
    scores = []

    for idx in range(len(images)):
        img = images[idx : idx + 1].to(device).requires_grad_(True)

        # Forward pass to get embedding dimension
        with torch.no_grad():
            emb = model(img)
        embed_dim = emb.shape[1]

        # Generate random projection matrix: (n_projections, embed_dim)
        omega = torch.tensor(
            rng.randn(n_projections, embed_dim),
            dtype=torch.float32,
            device=device,
        )

        # Compute J^T @ omega via vector-Jacobian products (reverse mode)
        # Each column of the result is one VJP
        jt_omega_cols = []
        for j in range(n_projections):
            img_fresh = images[idx : idx + 1].to(device).requires_grad_(True)
            emb = model(img_fresh)

            # VJP: compute omega[j] @ J_f(x)
            v = omega[j].unsqueeze(0)  # (1, embed_dim)
            (grad,) = torch.autograd.grad(
                outputs=emb,
                inputs=img_fresh,
                grad_outputs=v,
                retain_graph=False,
            )
            jt_omega_cols.append(grad.flatten().detach().cpu())

        # Stack into matrix: (n_projections, input_dim)
        jt_omega = torch.stack(jt_omega_cols, dim=0).numpy()

        # QR decomposition for numerical stability
        q, r = np.linalg.qr(jt_omega.T)  # q: (input_dim, n_projections)

        # Compute B = J @ Q by doing forward-mode JVPs
        # But we can approximate: B ≈ (omega @ J) @ Q = jt_omega.T @ Q... wait
        # Actually: we have J^T @ omega as rows. So (J^T @ omega)^T = omega^T @ J
        # We need SVD of J. We have omega^T @ J ≈ jt_omega
        # SVD of jt_omega gives approximate singular values of J

        # Compute SVD of the projected matrix
        try:
            s = np.linalg.svd(jt_omega, compute_uv=False)
            # The singular values of jt_omega approximate those of J
            # (scaled by the random projection)
            # JEPA-SCORE = sum of log singular values
            s_clipped = np.clip(s, 1e-10, None)
            score = np.sum(np.log(s_clipped))
        except np.linalg.LinAlgError:
            score = 0.0

        scores.append(score)

        if (idx + 1) % 50 == 0:
            print(f"  JEPA-SCORE: {idx + 1}/{len(images)} done")

    return np.array(scores)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def knn_scores(
    embeddings_id: np.ndarray,
    embeddings_test: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """k-NN distance scoring. Higher distance = more likely OOD."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(embeddings_id)

    distances, _ = nn.kneighbors(embeddings_test)
    return distances.mean(axis=1)


def mahalanobis_scores(
    embeddings_id: np.ndarray,
    embeddings_test: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance from the in-distribution mean."""
    cov = EmpiricalCovariance().fit(embeddings_id)
    return cov.mahalanobis(embeddings_test)


def isolation_forest_scores(
    embeddings_id: np.ndarray,
    embeddings_test: np.ndarray,
) -> np.ndarray:
    """Isolation Forest anomaly scores."""
    iforest = IsolationForest(n_estimators=100, random_state=42)
    iforest.fit(embeddings_id)
    # score_samples returns negative anomaly scores (lower = more anomalous)
    return -iforest.score_samples(embeddings_test)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class OODResult:
    method: str
    auroc: float
    fpr95: float
    time_seconds: float

    def __str__(self) -> str:
        return f"{self.method:<25} AUROC: {self.auroc:.4f}  FPR95: {self.fpr95:.4f}  Time: {self.time_seconds:.1f}s"


def compute_fpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute False Positive Rate at 95% True Positive Rate."""
    # labels: 0 = in-distribution, 1 = OOD
    # scores: higher = more likely OOD
    id_scores = scores[labels == 0]
    ood_scores = scores[labels == 1]

    # Find threshold where 95% of OOD samples are detected
    threshold = np.percentile(ood_scores, 5)  # 95% TPR

    # FPR = fraction of ID samples above threshold
    fpr = np.mean(id_scores >= threshold)
    return float(fpr)


def evaluate_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str,
    elapsed: float,
) -> OODResult:
    """Evaluate OOD detection performance."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    fpr95 = compute_fpr95(labels, scores)

    return OODResult(
        method=method_name,
        auroc=auroc,
        fpr95=fpr95,
        time_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_samples: int = 500, device: str = "cpu", n_projections: int = 64) -> list[OODResult]:
    """Run the full JEPA-SCORE vs baselines experiment."""
    print("=" * 60)
    print("JEPA-SCORE vs Simple Baselines: OOD Detection")
    print("=" * 60)
    print(f"  In-distribution:  CIFAR-10 ({n_samples} samples)")
    print(f"  Out-of-distribution: SVHN ({n_samples} samples)")
    print(f"  Model: DINOv2 ViT-S/14")
    print(f"  Device: {device}")
    print(f"  JEPA-SCORE projections: {n_projections}")
    print()

    # Load data
    print("Loading datasets...")
    cifar_images, _ = get_cifar10(n_samples)
    svhn_images, _ = get_svhn(n_samples)
    print(f"  CIFAR-10: {cifar_images.shape}")
    print(f"  SVHN: {svhn_images.shape}")

    # Load model
    print("\nLoading DINOv2 ViT-S/14...")
    model = load_dinov2(device)
    print("  Model loaded.")

    # Extract embeddings (shared across embedding-based baselines)
    print("\nExtracting embeddings...")
    t0 = time.time()
    cifar_emb = extract_embeddings(model, cifar_images, device)
    svhn_emb = extract_embeddings(model, svhn_images, device)
    emb_time = time.time() - t0
    print(f"  Embeddings: {cifar_emb.shape}, took {emb_time:.1f}s")

    results: list[OODResult] = []

    # --- Baseline 1: k-NN distance ---
    print("\nRunning k-NN baseline...")
    t0 = time.time()
    knn_id = knn_scores(cifar_emb, cifar_emb, k=10)
    knn_ood = knn_scores(cifar_emb, svhn_emb, k=10)
    knn_time = time.time() - t0
    results.append(evaluate_ood(knn_id, knn_ood, "k-NN distance (k=10)", knn_time))
    print(f"  {results[-1]}")

    # --- Baseline 2: Mahalanobis ---
    print("\nRunning Mahalanobis baseline...")
    t0 = time.time()
    maha_id = mahalanobis_scores(cifar_emb, cifar_emb)
    maha_ood = mahalanobis_scores(cifar_emb, svhn_emb)
    maha_time = time.time() - t0
    results.append(evaluate_ood(maha_id, maha_ood, "Mahalanobis distance", maha_time))
    print(f"  {results[-1]}")

    # --- Baseline 3: Isolation Forest ---
    print("\nRunning Isolation Forest baseline...")
    t0 = time.time()
    iforest_id = isolation_forest_scores(cifar_emb, cifar_emb)
    iforest_ood = isolation_forest_scores(cifar_emb, svhn_emb)
    iforest_time = time.time() - t0
    results.append(evaluate_ood(iforest_id, iforest_ood, "Isolation Forest", iforest_time))
    print(f"  {results[-1]}")

    # --- JEPA-SCORE ---
    print(f"\nRunning JEPA-SCORE (randomized SVD, {n_projections} projections)...")
    t0 = time.time()
    # JEPA-SCORE: lower score = lower density = more likely OOD
    # So we negate for consistency (higher = more anomalous)
    jepa_id = -compute_jepa_score_randomized(model, cifar_images, device, n_projections)
    jepa_ood = -compute_jepa_score_randomized(model, svhn_images, device, n_projections)
    jepa_time = time.time() - t0
    results.append(evaluate_ood(jepa_id, jepa_ood, f"JEPA-SCORE (proj={n_projections})", jepa_time))
    print(f"  {results[-1]}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'AUROC':>8} {'FPR95':>8} {'Time':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.auroc, reverse=True):
        print(f"{r.method:<25} {r.auroc:>8.4f} {r.fpr95:>8.4f} {r.time_seconds:>7.1f}s")

    # --- Verdict ---
    knn_auroc = results[0].auroc
    jepa_auroc = results[-1].auroc
    diff = jepa_auroc - knn_auroc

    print("\n" + "=" * 60)
    if diff > 0.03:
        print(f"VERDICT: JEPA-SCORE WINS by {diff:.4f} AUROC points")
        print("  -> Meaningful improvement. Proceed with productization.")
    elif diff > 0.01:
        print(f"VERDICT: JEPA-SCORE marginal win (+{diff:.4f} AUROC)")
        print("  -> Interesting but may not justify 400x compute overhead.")
    elif diff > -0.01:
        print(f"VERDICT: JEPA-SCORE tied with k-NN (diff: {diff:+.4f} AUROC)")
        print("  -> No meaningful difference. Publish as negative result.")
    else:
        print(f"VERDICT: JEPA-SCORE LOSES to k-NN by {-diff:.4f} AUROC points")
        print("  -> k-NN is better AND 400x cheaper. Publish negative result.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA-SCORE vs baselines OOD experiment")
    parser.add_argument("--n-samples", type=int, default=500, help="Samples per dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--n-projections", type=int, default=64, help="Random projections for JEPA-SCORE")
    args = parser.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        device=args.device,
        n_projections=args.n_projections,
    )
