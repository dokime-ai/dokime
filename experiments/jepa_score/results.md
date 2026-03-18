# JEPA-SCORE Experiment Results
## March 18, 2026

### Setup
- **In-distribution:** CIFAR-10 test set (500 samples)
- **Out-of-distribution:** SVHN test set (500 samples)
- **Model:** DINOv2 ViT-S/14 (21M params, Apache 2.0)
- **Hardware:** NVIDIA RTX 3090 (24GB VRAM)
- **JEPA-SCORE config:** 64 random projections, randomized SVD

### Results

| Method | AUROC | FPR95 | Time |
|--------|-------|-------|------|
| Mahalanobis distance | **1.0000** | **0.0000** | 0.2s |
| k-NN distance (k=10) | 0.9652 | 0.1940 | 0.0s |
| JEPA-SCORE (proj=64) | 0.7285 | 0.7360 | 1925.5s |
| Isolation Forest | 0.4387 | 0.9120 | 0.2s |

### Verdict

**JEPA-SCORE loses to k-NN by 23.67 AUROC points and is ~10,000x slower.**

Both Mahalanobis and k-NN achieve near-perfect OOD detection using the same
DINOv2 embeddings, in under 1 second. JEPA-SCORE, which requires computing
the encoder Jacobian via 64 backward passes per sample, takes 32 minutes
and achieves significantly worse results.

### Implications

1. **For the Dokime product:** JEPA-SCORE is not viable as a product feature.
   k-NN anomaly scoring (already implemented in Dokime) is both better and faster.

2. **For the research community:** This is the first independent evaluation of
   JEPA-SCORE (arXiv:2510.05949, NeurIPS 2025) against standard baselines.
   The original paper contained no baseline comparisons. Our results suggest
   the Jacobian-based density estimation does not outperform simple distance-based
   methods for OOD detection on pretrained vision encoders.

3. **Caveats:**
   - 500 samples per class (larger study needed for publication)
   - Only one encoder (DINOv2 ViT-S/14) tested
   - Randomized SVD approximation may lose accuracy vs full Jacobian
   - CIFAR-10 vs SVHN is a well-separated OOD pair (results may differ on harder pairs)

### Next steps
- Expand to 5,000+ samples for statistical significance
- Test on harder OOD pairs (CIFAR-10 vs CIFAR-100, near-OOD)
- Test with full Jacobian (not randomized) on a small subset
- Write up as arXiv preprint regardless of outcome
