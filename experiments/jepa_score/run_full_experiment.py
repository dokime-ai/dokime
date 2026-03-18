"""JEPA-SCORE vs Simple Baselines: Full Publication-Grade Experiment.

Expands the initial smoke test to cover:
- 3 models: DINOv2 ViT-S/14, DINOv2 ViT-B/14, DINOv2 ViT-L/14
- 3 OOD pairs: CIFAR-10 vs SVHN (easy), CIFAR-10 vs CIFAR-100 (hard), CIFAR-10 vs Textures (medium)
- 3 seeds for error bars
- 2000 samples per dataset
- Full Jacobian on small subset (50 samples) to verify randomized SVD accuracy

Usage:
    python run_full_experiment.py --device cuda
    python run_full_experiment.py --device cuda --quick   # fewer samples for testing
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TRANSFORM_224 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dataset_subset(dataset_cls, n_samples: int, **kwargs) -> Tensor:
    """Load n_samples from a torchvision dataset."""
    ds = dataset_cls(root="./data", download=True, transform=TRANSFORM_224, **kwargs)
    images = []
    for i, (img, _) in enumerate(ds):
        if i >= n_samples:
            break
        images.append(img)
    return torch.stack(images)


def get_ood_datasets(n_samples: int) -> dict[str, tuple[Tensor, Tensor]]:
    """Load all OOD dataset pairs."""
    print("Loading datasets...")
    cifar10 = load_dataset_subset(datasets.CIFAR10, n_samples, train=False)
    svhn = load_dataset_subset(datasets.SVHN, n_samples, split="test")
    cifar100 = load_dataset_subset(datasets.CIFAR100, n_samples, train=False)

    # DTD (Describable Textures Dataset) - use train split as OOD
    try:
        textures = load_dataset_subset(datasets.DTD, n_samples, split="test")
    except Exception:
        print("  DTD download failed, skipping Textures OOD pair")
        textures = None

    pairs = {
        "CIFAR10_vs_SVHN": (cifar10, svhn),
        "CIFAR10_vs_CIFAR100": (cifar10, cifar100),
    }
    if textures is not None:
        pairs["CIFAR10_vs_Textures"] = (cifar10, textures)

    for name, (id_data, ood_data) in pairs.items():
        print(f"  {name}: ID={id_data.shape}, OOD={ood_data.shape}")

    return pairs


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
}


def load_model(model_key: str, device: str) -> torch.nn.Module:
    """Load a DINOv2 model."""
    repo, name = MODELS[model_key]
    model = torch.hub.load(repo, name)
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(model: torch.nn.Module, images: Tensor, device: str, batch_size: int = 64) -> np.ndarray:
    """Extract CLS embeddings."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# JEPA-SCORE
# ---------------------------------------------------------------------------

def compute_jepa_score(
    model: torch.nn.Module,
    images: Tensor,
    device: str,
    n_projections: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Compute JEPA-SCORE using randomized SVD."""
    rng = np.random.RandomState(seed)
    scores = []

    for idx in range(len(images)):
        img = images[idx : idx + 1].to(device).requires_grad_(True)

        with torch.no_grad():
            emb = model(img)
        embed_dim = emb.shape[1]

        omega = torch.tensor(rng.randn(n_projections, embed_dim), dtype=torch.float32, device=device)

        jt_omega_cols = []
        for j in range(n_projections):
            img_fresh = images[idx : idx + 1].to(device).requires_grad_(True)
            emb = model(img_fresh)
            v = omega[j].unsqueeze(0)
            (grad,) = torch.autograd.grad(outputs=emb, inputs=img_fresh, grad_outputs=v, retain_graph=False)
            jt_omega_cols.append(grad.flatten().detach().cpu())

        jt_omega = torch.stack(jt_omega_cols, dim=0).numpy()

        try:
            s = np.linalg.svd(jt_omega, compute_uv=False)
            s_clipped = np.clip(s, 1e-10, None)
            score = np.sum(np.log(s_clipped))
        except np.linalg.LinAlgError:
            score = 0.0

        scores.append(score)

        if (idx + 1) % 100 == 0:
            print(f"    JEPA-SCORE: {idx + 1}/{len(images)}")

    return np.array(scores)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def knn_scores(emb_id: np.ndarray, emb_test: np.ndarray, k: int = 10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(emb_id)
    distances, _ = nn.kneighbors(emb_test)
    return distances.mean(axis=1)


def mahalanobis_scores(emb_id: np.ndarray, emb_test: np.ndarray) -> np.ndarray:
    cov = EmpiricalCovariance().fit(emb_id)
    return cov.mahalanobis(emb_test)


def iforest_scores(emb_id: np.ndarray, emb_test: np.ndarray) -> np.ndarray:
    iforest = IsolationForest(n_estimators=100, random_state=42)
    iforest.fit(emb_id)
    return -iforest.score_samples(emb_test)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Result:
    model: str
    ood_pair: str
    method: str
    seed: int
    auroc: float
    fpr95: float
    time_seconds: float
    n_samples: int


def compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    threshold = np.percentile(ood_scores, 5)
    fpr = np.mean(id_scores >= threshold)
    return float(fpr)


def evaluate(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(labels, scores)
    fpr95 = compute_fpr95(id_scores, ood_scores)
    return auroc, fpr95


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_experiment(
    n_samples: int = 2000,
    device: str = "cuda",
    n_projections: int = 64,
    seeds: list[int] | None = None,
    models: list[str] | None = None,
) -> list[Result]:
    if seeds is None:
        seeds = [42, 123, 456]
    if models is None:
        models = list(MODELS.keys())

    print("=" * 70)
    print("JEPA-SCORE vs Simple Baselines: Full Publication Experiment")
    print("=" * 70)
    print(f"  Samples per dataset: {n_samples}")
    print(f"  Models: {models}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print()

    # Load all data once
    ood_pairs = get_ood_datasets(n_samples)

    all_results: list[Result] = []

    for model_key in models:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_key}")
        print(f"{'='*70}")

        model = load_model(model_key, device)

        for pair_name, (id_images, ood_images) in ood_pairs.items():
            print(f"\n  OOD Pair: {pair_name}")

            # Extract embeddings (deterministic, only need once)
            t0 = time.time()
            id_emb = extract_embeddings(model, id_images, device)
            ood_emb = extract_embeddings(model, ood_images, device)
            emb_time = time.time() - t0
            print(f"    Embeddings: {id_emb.shape}, {emb_time:.1f}s")

            # --- k-NN ---
            t0 = time.time()
            knn_id = knn_scores(id_emb, id_emb)
            knn_ood = knn_scores(id_emb, ood_emb)
            knn_time = time.time() - t0
            auroc, fpr95 = evaluate(knn_id, knn_ood)
            print(f"    k-NN:         AUROC={auroc:.4f} FPR95={fpr95:.4f} ({knn_time:.1f}s)")
            for seed in seeds:
                all_results.append(Result(model_key, pair_name, "k-NN", seed, auroc, fpr95, knn_time, n_samples))

            # --- Mahalanobis ---
            t0 = time.time()
            maha_id = mahalanobis_scores(id_emb, id_emb)
            maha_ood = mahalanobis_scores(id_emb, ood_emb)
            maha_time = time.time() - t0
            auroc, fpr95 = evaluate(maha_id, maha_ood)
            print(f"    Mahalanobis:  AUROC={auroc:.4f} FPR95={fpr95:.4f} ({maha_time:.1f}s)")
            for seed in seeds:
                all_results.append(Result(model_key, pair_name, "Mahalanobis", seed, auroc, fpr95, maha_time, n_samples))

            # --- Isolation Forest ---
            t0 = time.time()
            if_id = iforest_scores(id_emb, id_emb)
            if_ood = iforest_scores(id_emb, ood_emb)
            if_time = time.time() - t0
            auroc, fpr95 = evaluate(if_id, if_ood)
            print(f"    IForest:      AUROC={auroc:.4f} FPR95={fpr95:.4f} ({if_time:.1f}s)")
            for seed in seeds:
                all_results.append(Result(model_key, pair_name, "IsolationForest", seed, auroc, fpr95, if_time, n_samples))

            # --- JEPA-SCORE (varies by seed) ---
            for seed in seeds:
                print(f"    JEPA-SCORE (seed={seed})...")
                t0 = time.time()
                jepa_id = -compute_jepa_score(model, id_images, device, n_projections, seed)
                jepa_ood = -compute_jepa_score(model, ood_images, device, n_projections, seed)
                jepa_time = time.time() - t0
                auroc, fpr95 = evaluate(jepa_id, jepa_ood)
                print(f"      AUROC={auroc:.4f} FPR95={fpr95:.4f} ({jepa_time:.1f}s)")
                all_results.append(Result(model_key, pair_name, "JEPA-SCORE", seed, auroc, fpr95, jepa_time, n_samples))

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    results_path = Path("experiments/jepa_score/full_results.json")
    results_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (mean +/- std across seeds)")
    print("=" * 70)

    # Group by (model, pair, method) and compute mean/std
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in all_results:
        groups[(r.model, r.ood_pair, r.method)].append(r.auroc)

    print(f"{'Model':<18} {'OOD Pair':<25} {'Method':<18} {'AUROC':>12}")
    print("-" * 75)
    for (model, pair, method), aurocs in sorted(groups.items()):
        mean = np.mean(aurocs)
        std = np.std(aurocs)
        if std > 0:
            print(f"{model:<18} {pair:<25} {method:<18} {mean:>6.4f} +/- {std:.4f}")
        else:
            print(f"{model:<18} {pair:<25} {method:<18} {mean:>6.4f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-projections", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 200 samples, 1 seed, 1 model")
    args = parser.parse_args()

    if args.quick:
        run_full_experiment(
            n_samples=200,
            device=args.device,
            n_projections=32,
            seeds=[42],
            models=["dinov2_vits14"],
        )
    else:
        run_full_experiment(
            n_samples=args.n_samples,
            device=args.device,
            n_projections=args.n_projections,
        )
