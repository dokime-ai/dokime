"""JEPA-SCORE vs Baselines — GPU-optimized. SVD runs on GPU via torch.linalg.svdvals."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms

SEP = "=" * 70

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_subset(cls, n, **kw):
    ds = cls(root="./data", download=True, transform=TRANSFORM, **kw)
    imgs = []
    for i, (img, _) in enumerate(ds):
        if i >= n:
            break
        imgs.append(img)
    return torch.stack(imgs)


def get_datasets(n):
    print("Loading datasets...", flush=True)
    c10 = load_subset(datasets.CIFAR10, n, train=False)
    svhn = load_subset(datasets.SVHN, n, split="test")
    c100 = load_subset(datasets.CIFAR100, n, train=False)
    try:
        dtd = load_subset(datasets.DTD, n, split="test")
    except Exception:
        dtd = None
    pairs = {"CIFAR10_vs_SVHN": (c10, svhn), "CIFAR10_vs_CIFAR100": (c10, c100)}
    if dtd is not None:
        pairs["CIFAR10_vs_Textures"] = (c10, dtd)
    for k, (a, b) in pairs.items():
        print(f"  {k}: {a.shape} vs {b.shape}", flush=True)
    return pairs


MODELS = {
    "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
}


def load_model(key, device):
    repo, name = MODELS[key]
    m = torch.hub.load(repo, name)
    m.to(device).eval()
    return m


def extract_emb(model, imgs, device, bs=64):
    embs = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            embs.append(model(imgs[i : i + bs].to(device)).cpu().numpy())
    return np.concatenate(embs)


def jepa_score_gpu(model, imgs, device, n_proj=64, seed=42):
    """JEPA-SCORE with GPU-accelerated SVD. No CPU transfers until final score."""
    rng = torch.Generator(device=device).manual_seed(seed)
    scores = []
    for idx in range(len(imgs)):
        with torch.no_grad():
            emb = model(imgs[idx : idx + 1].to(device))
        edim = emb.shape[1]
        omega = torch.randn(n_proj, edim, device=device, generator=rng)

        jt_cols = []
        for j in range(n_proj):
            x = imgs[idx : idx + 1].to(device).requires_grad_(True)
            out = model(x)
            (g,) = torch.autograd.grad(out, x, grad_outputs=omega[j].unsqueeze(0), retain_graph=False)
            jt_cols.append(g.flatten().detach())

        jt = torch.stack(jt_cols, dim=0)  # stays on GPU
        s = torch.linalg.svdvals(jt)  # GPU SVD
        score = s.clamp(min=1e-10).log().sum().item()
        scores.append(score)

        if (idx + 1) % 100 == 0:
            print(f"    JEPA-SCORE: {idx + 1}/{len(imgs)}", flush=True)
    return np.array(scores)


def knn_scores(eid, etest, k=10):
    return NearestNeighbors(n_neighbors=k, metric="cosine").fit(eid).kneighbors(etest)[0].mean(axis=1)


def maha_scores(eid, etest):
    return EmpiricalCovariance().fit(eid).mahalanobis(etest)


def iforest_scores(eid, etest):
    return -IsolationForest(n_estimators=100, random_state=42).fit(eid).score_samples(etest)


@dataclass
class R:
    model: str
    ood: str
    method: str
    seed: int
    auroc: float
    fpr95: float
    time_s: float
    n: int


def fpr95(id_s, ood_s):
    thr = np.percentile(ood_s, 5)
    return float(np.mean(id_s >= thr))


def evaluate(id_s, ood_s):
    labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    scores = np.concatenate([id_s, ood_s])
    return roc_auc_score(labels, scores), fpr95(id_s, ood_s)


def run(n=2000, device="cuda", n_proj=64, seeds=None, models=None):
    if seeds is None:
        seeds = [42, 123, 456]
    if models is None:
        models = list(MODELS.keys())
    print(SEP)
    print("JEPA-SCORE vs Baselines: GPU-Optimized Full Experiment")
    print(SEP)
    print(f"  Samples: {n}, Models: {models}, Seeds: {seeds}, Device: {device}", flush=True)

    pairs = get_datasets(n)
    results = []

    for mk in models:
        print(f"\n{SEP}\nMODEL: {mk}\n{SEP}", flush=True)
        model = load_model(mk, device)

        for pn, (id_img, ood_img) in pairs.items():
            print(f"\n  {pn}", flush=True)
            t0 = time.time()
            id_e = extract_emb(model, id_img, device)
            ood_e = extract_emb(model, ood_img, device)
            print(f"    Embeddings: {id_e.shape}, {time.time()-t0:.1f}s", flush=True)

            # k-NN
            t0 = time.time()
            ki, ko = knn_scores(id_e, id_e), knn_scores(id_e, ood_e)
            kt = time.time() - t0
            a, f = evaluate(ki, ko)
            print(f"    k-NN:        AUROC={a:.4f} FPR95={f:.4f} ({kt:.1f}s)", flush=True)
            for s in seeds:
                results.append(R(mk, pn, "k-NN", s, a, f, kt, n))

            # Mahalanobis
            t0 = time.time()
            mi, mo = maha_scores(id_e, id_e), maha_scores(id_e, ood_e)
            mt = time.time() - t0
            a, f = evaluate(mi, mo)
            print(f"    Mahalanobis: AUROC={a:.4f} FPR95={f:.4f} ({mt:.1f}s)", flush=True)
            for s in seeds:
                results.append(R(mk, pn, "Mahalanobis", s, a, f, mt, n))

            # IForest
            t0 = time.time()
            ii, io_ = iforest_scores(id_e, id_e), iforest_scores(id_e, ood_e)
            it = time.time() - t0
            a, f = evaluate(ii, io_)
            print(f"    IForest:     AUROC={a:.4f} FPR95={f:.4f} ({it:.1f}s)", flush=True)
            for s in seeds:
                results.append(R(mk, pn, "IsolationForest", s, a, f, it, n))

            # JEPA-SCORE per seed
            for seed in seeds:
                print(f"    JEPA-SCORE (seed={seed})...", flush=True)
                t0 = time.time()
                ji = -jepa_score_gpu(model, id_img, device, n_proj, seed)
                jo = -jepa_score_gpu(model, ood_img, device, n_proj, seed)
                jt = time.time() - t0
                a, f = evaluate(ji, jo)
                print(f"      AUROC={a:.4f} FPR95={f:.4f} ({jt:.1f}s)", flush=True)
                results.append(R(mk, pn, "JEPA-SCORE", seed, a, f, jt, n))

        del model
        torch.cuda.empty_cache()

    # Save
    p = Path("experiments/jepa_score/full_results_gpu.json")
    p.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nSaved to {p}", flush=True)

    # Summary
    print(f"\n{SEP}\nAGGREGATED RESULTS (mean +/- std)\n{SEP}", flush=True)
    groups = defaultdict(list)
    for r in results:
        groups[(r.model, r.ood, r.method)].append(r.auroc)
    print(f"{'Model':<18} {'OOD':<25} {'Method':<18} {'AUROC':>12}", flush=True)
    print("-" * 75, flush=True)
    for (m, p, met), aurocs in sorted(groups.items()):
        mn, st = np.mean(aurocs), np.std(aurocs)
        if st > 0:
            print(f"{m:<18} {p:<25} {met:<18} {mn:>6.4f} +/- {st:.4f}", flush=True)
        else:
            print(f"{m:<18} {p:<25} {met:<18} {mn:>6.4f}", flush=True)
    print(SEP, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-projections", type=int, default=64)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        run(200, a.device, 32, [42], ["dinov2_vits14"])
    else:
        run(a.n_samples, a.device, a.n_projections)
