# Dokime Messy Dataset Audit Results

**Date:** 2026-03-19
**Tool:** dokime score v0.x (1000-sample streaming audit)
**Goal:** Find HuggingFace datasets that score below A to demonstrate what bad training data looks like.

## Executive Summary

We audited 18 HuggingFace datasets with dokime score, streaming 1000 samples from each.
The majority of well-known English text datasets score A (90+). To find non-A scores, we had to look at:

1. **Source code** datasets (not prose)
2. **Non-English CJK text** (exposes English-centric bias in word-based metrics)
3. **Raw web crawl** with structural/metadata contamination

**Key finding:** The current scorer is generous for English prose. Even C4 en.noclean (the UNFILTERED version of C4) still scores A (94.3).

## Full Results (sorted by score)

| Rank | Dataset | Grade | Score |
|------|---------|-------|-------|
| 1 | FineWeb (HuggingFaceFW/fineweb) | **A** | 99.8 |
| 2 | C4 en (allenai/c4) | **A** | 99.7 |
| 3 | SlimOrca-Dedup (Open-Orca/SlimOrca-Dedup) | **A** | 99.7 |
| 4 | Falcon RefinedWeb (tiiuae/falcon-refinedweb) | **A** | 99.3 |
| 5 | Wikipedia en (wikimedia/wikipedia) | **A** | 99.0 |
| 6 | OpenHermes-2.5 (teknium/OpenHermes-2.5) | **A** | 98.6 |
| 7 | Pile Uncopyrighted (monology/pile-uncopyrighted) | **A** | 96.7 |
| 8 | Open-Platypus (garage-bAInd/Open-Platypus) | **A** | 95.4 |
| 9 | C4 en.noclean (allenai/c4 noclean) | **A** | 94.3 |
| 10 | Tweet Eval (tweet_eval) | **A** | 93.3 |
| 11 | Claire French (OpenLLM-France/Claire-Dialogue-French-0.1) | **B+** | 89.5 |
| 12 | Codeparrot-clean (codeparrot/codeparrot-clean) | **B+** | 88.0 |
| 13 | Jupyter Code (codeparrot/github-jupyter-text-code-pairs) | **B** | 78.9 |
| 14 | Wikipedia th (wikimedia/wikipedia th) | **C+** | 56.9 |
| 15 | Wikipedia ja (wikimedia/wikipedia ja) | **C+** | 51.4 |
| 16 | Wikipedia zh (wikimedia/wikipedia zh) | **C+** | 50.1 |
| 17 | crawl-dataset (philschmid/crawl-dataset) | **C+** | 50.0 |

## The 3 Best Non-A Real HuggingFace Datasets

### 1. codeparrot/github-jupyter-text-code-pairs -- Grade B (78.9/100)

Raw Jupyter notebook code cells from GitHub. 285/1000 docs very short (<10 words),
27 high special char ratio, 2 low-entropy repetitive docs, 1 exact duplicate.
Distribution: 58% excellent, 26% good, 16% medium, 0.1% poor.

### 2. codeparrot/codeparrot-clean -- Grade B+ (88.0/100)

Cleaned Python source code from GitHub. 6 high special char docs, 132 good (0.6-0.8),
9 medium (0.3-0.6). Code fails the alpha-ratio and word-length sweet spots for prose.

### 3. philschmid/crawl-dataset -- Grade C+ (50.0/100)

Web crawl of HuggingFace docs. Text field contains JSON-stringified dicts with title,
URL, and markdown -- structural metadata, not clean text. 0% excellent, 77% medium.
This is a real data engineering mistake: storing structured data as raw strings.

## Non-English Datasets Expose Scorer Limitations

| Dataset | Grade | Score | Why |
|---------|-------|-------|-----|
| Wikipedia Thai | C+ | 56.9 | No spaces between words |
| Wikipedia Japanese | C+ | 51.4 | CJK: long words, low alpha ratio |
| Wikipedia Chinese | C+ | 50.1 | Same CJK issue |
| Wikipedia Arabic | A | 98.9 | Arabic uses spaces like English |

## Key Takeaways

1. Most curated English text datasets score A. The scorer is too lenient for prose.
2. Code datasets score B/B+. Correct -- code IS structurally different from prose.
3. Non-English CJK datasets score C+. Scorer limitation, not data quality issue.
4. Structural contamination (JSON in text) correctly scores C+. Real problem.
5. Even C4 en.noclean (unfiltered!) scores A (94.3). Need more signals.

## Scorer Improvement Ideas

- N-gram repetition detection (boilerplate, nav menus)
- Language detection (mixed-language contamination)
- URL/HTML tag density (incomplete extraction)
- Near-duplicate detection (MinHash/SimHash)
- Perplexity scoring (KenLM)
- Unicode script mixing (mojibake detection)

## Files

All JSONL samples in blog/data/. Full JSON results: blog/data/all_audit_results.json
