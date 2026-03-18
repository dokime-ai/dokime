# JEPA-SCORE Validation Experiment

First independent evaluation of JEPA-SCORE (arXiv:2510.05949, NeurIPS 2025)
for out-of-distribution detection, benchmarked against simple baselines.

## What this tests

The JEPA-SCORE paper claims that the sum of log-singular-values of a JEPA encoder's
Jacobian estimates data density, enabling outlier/OOD detection. This experiment
tests whether JEPA-SCORE actually beats simpler methods on real data.

## Setup

```bash
pip install torch torchvision scikit-learn
python run_experiment.py
```

## Methods compared

1. **JEPA-SCORE** — Randomized SVD of encoder Jacobian (Halko-Martinsson-Tropp)
2. **k-NN distance** — Average distance to k nearest neighbors in embedding space
3. **Mahalanobis distance** — Distance from class-conditional Gaussian in embedding space
4. **Isolation Forest** — Tree-based anomaly detection on embeddings

## Benchmark

- **In-distribution:** CIFAR-10 test set
- **Out-of-distribution:** SVHN test set
- **Model:** DINOv2 ViT-S/14 (smallest variant, Apache 2.0)
- **Metrics:** AUROC, FPR95

## Kill criteria

If JEPA-SCORE fails to beat k-NN by >1 AUROC point → negative result, still publishable.
