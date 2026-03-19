# We Scored 14 Training Datasets -- Here's What We Found

*March 2026 | Andrew Morgan*

We built a data quality scorer, tested it on 14 HuggingFace datasets, discovered it was too lenient to be useful, rebuilt it with signals from the academic literature, and now it actually works. This is the story of that process and what we learned about the data inside some of the most popular training corpora.

---

## TL;DR

Our v2 scorer uses 12 signals drawn from Gopher, C4, and FineWeb to grade datasets on a 0-100 scale. Here is how 14 well-known HuggingFace datasets scored:

| Dataset | Grade | Score |
|---------|-------|-------|
| FineWeb | A | 99.0% |
| OpenWebText | A | 98.9% |
| C4 en (filtered) | A | 97.8% |
| Wikipedia (raw) | A | 97.9% |
| SlimOrca | A | 96.7% |
| Falcon RefinedWeb | A | 96.0% |
| The Pile | A | 92.8% |
| Open Platypus | A | 91.9% |
| OpenHermes | A | 91.2% |
| codeparrot-clean | B+ | 84.1% |
| tweet_eval | B | 79.9% |
| C4 en.noclean (RAW) | B | 78.8% |
| crawl-dataset | B- | 69.7% |
| jupyter-pairs | C+ | 59.9% |

The key result: **C4 en.noclean, the unfiltered version of C4, dropped from an A (94.3%) under our old scorer to a B (78.8%) under the new one.** That 15-point drop is the difference between a scorer that tells you everything is fine and one that tells you where the problems are.

---

## The Problem: Our V1 Scorer Was Useless

Dokime's first quality scorer used four signals: character entropy, alphabetic ratio, word count, and average word length. These are reasonable features. They catch empty documents, binary data, and strings of random characters. What they do not catch is the kind of low-quality content that actually contaminates real training data -- boilerplate text, repetitive content, poorly extracted web pages, and navigation menus.

We discovered this the hard way. We ran the v1 scorer against 14 HuggingFace datasets and every single one scored an A. C4 en.noclean, which is the *unfiltered* version of C4 -- the raw Common Crawl text before any cleaning -- scored 94.3%. OpenWebText scored 99.8%. C4 en (the filtered version) scored 99.7%.

The scorer was not discriminating between filtered and unfiltered data. If your quality tool gives the same grade to raw Common Crawl and a carefully curated dataset, it is not a quality tool. It is a rubber stamp.

We needed better signals.

---

## What We Added and Why

We went to the literature. Three papers define the modern standard for heuristic data quality filtering, and each contributes signals that our v1 was missing.

### Gopher (Rae et al. 2021)

The Gopher paper ([arXiv:2112.11446](https://arxiv.org/abs/2112.11446)) introduced a comprehensive set of heuristic filters for web-scraped text. The signals we adopted:

- **Word count bounds.** Documents that are too short or too long are disproportionately low quality.
- **Average word length.** Catches encoding artifacts, URL dumps, and content where tokenization has gone wrong.
- **Stop word ratio.** Natural English text contains common function words ("the", "and", "of"). Content with very few stop words is often keyword-stuffed, machine-generated, or not natural language at all.
- **Duplicate line ratio.** The fraction of lines in a document that are exact copies of another line in the same document. Navigation menus, repeated headers, cookie banners, and template footers all produce duplicate lines.
- **Top n-gram frequency.** If a small number of n-grams dominate a document, it is likely repetitive boilerplate rather than diverse natural text.

### C4 / T5 (Raffel et al. 2020)

The C4 dataset paper ([arXiv:1910.10683](https://arxiv.org/abs/1910.10683)) defined the filtering pipeline used to produce C4 from Common Crawl. The signals we adopted:

- **Sentence count.** Documents with very few sentences are often not prose -- they are titles, captions, or metadata.
- **Boilerplate detection.** Patterns like "terms of use", "cookie policy", "JavaScript required", and similar boilerplate that indicate the document is a website shell rather than content.
- **Terminal punctuation ratio.** The fraction of sentences that end with proper punctuation (period, question mark, exclamation point). Low terminal punctuation indicates lists, navigation elements, or fragmented text.

### FineWeb (Penedo et al. 2024)

The FineWeb paper ([arXiv:2406.17557](https://arxiv.org/abs/2406.17557)) refined and extended prior heuristics based on empirical analysis of what actually helps downstream model performance. The signals we adopted:

- **Line-level punctuation ratio.** The fraction of lines that end with punctuation. This is subtly different from C4's sentence-level metric and catches a different class of problems -- documents where lines are navigational elements, breadcrumbs, or structured data that lack natural sentence endings.
- **Short line ratio.** The fraction of lines below a character threshold. Documents dominated by short lines are typically menus, sidebars, or fragmented extractions.

All thresholds were verified against the [huggingface/datatrove](https://github.com/huggingface/datatrove) source code to ensure we matched the actual implementations, not just the paper descriptions. Papers sometimes leave out implementation details. The code does not.

Our v2 scorer combines all 12 signals into a composite score, weighted by their empirical importance for separating clean and dirty text.

---

## Before and After: C4 en.noclean

The most revealing test case is `allenai/c4` with the `en.noclean` config. This is raw Common Crawl text -- the input to C4's filtering pipeline, not the output. It is supposed to be messy. A useful quality scorer should reflect that.

**V1 scorer (4 signals):** Grade A, 94.3%. Indistinguishable from curated data.

**V2 scorer (12 signals):** Grade B, 78.8%. Correctly identifies significant quality issues.

Here is what the new signals found in C4 en.noclean:

| Signal | Flagged (%) | What it means |
|--------|-------------|---------------|
| High repetition (Gopher) | 59.4% | Duplicate n-grams, repeated phrases, boilerplate |
| Low punctuation (FineWeb) | 65.8% | Lines without natural sentence endings |
| Duplicate lines (Gopher) | 18.4% | Identical lines within the same document |
| Boilerplate content (C4) | 39.1% | Terms of service, cookie notices, JavaScript warnings |

Nearly 60% of documents in C4 en.noclean have high repetition rates. Two-thirds have low line-level punctuation. These are not edge cases -- they are the dominant failure modes of raw web crawl. Our v1 scorer missed all of them because it was only looking at character-level statistics, not document structure.

For comparison, C4 en (the filtered version) dropped from 99.7% under v1 to 97.8% under v2 -- a much smaller change. The filtered version still scores an A because the C4 pipeline already removes most of the content that our new signals detect. The 2-point drop reflects the small amount of borderline content that C4's filters let through.

---

## Full Results

Here is the complete v2 audit across all 14 datasets, sorted by score:

| Dataset | Grade | Score | Notes |
|---------|-------|-------|-------|
| FineWeb | A | 99.0% | State of the art. Penedo et al. applied the most thorough filtering pipeline in the literature. |
| OpenWebText | A | 98.9% | Reddit-upvoted content. Social curation acts as a natural quality gate. |
| Wikipedia (raw) | A | 97.9% | Encyclopedic prose. High structure, low noise. |
| C4 en (filtered) | A | 97.8% | Raffel et al.'s heuristic pipeline is effective. |
| SlimOrca | A | 96.7% | Curated instruction-following data. |
| Falcon RefinedWeb | A | 96.0% | TII's web-scale curation is solid. |
| The Pile | A | 92.8% | Intentionally diverse. Includes code and structured data that scores lower on prose metrics. |
| Open Platypus | A | 91.9% | STEM-focused instruction data. Some structured content pulls the score down. |
| OpenHermes | A | 91.2% | Synthetic + curated instruction data. Broader quality variance than SlimOrca. |
| codeparrot-clean | B+ | 84.1% | Source code is structurally different from prose. A B+ is correct, not a problem. |
| tweet_eval | B | 79.9% | Tweets are short, lack punctuation, and use non-standard language. Expected. |
| C4 en.noclean | B | 78.8% | Raw Common Crawl. Now correctly separated from its filtered counterpart. |
| crawl-dataset | B- | 69.7% | Web crawl with structural metadata contamination. Real data engineering issue. |
| jupyter-pairs | C+ | 59.9% | Jupyter notebook cell pairs. Dominated by code snippets and short fragments. |

The ordering tells a coherent story. The best-curated web datasets (FineWeb, OpenWebText) are at the top. Well-known filtered datasets (C4, Wikipedia, Falcon) cluster in the high 90s. Diverse or instruction-tuning datasets (The Pile, OpenHermes, Open Platypus) sit in the low 90s. Domain-specific data that is not prose (code, tweets) scores in the 80s. And genuinely messy data (raw crawl, notebook fragments) lands in the 60s-70s.

This is what a useful quality scorer should do: produce a ranking that matches your intuition about data quality, while giving you specific numbers to act on.

---

## What We Learned

**1. Four signals are not enough.** Character entropy, alpha ratio, word count, and average word length will catch catastrophically bad data -- binary files, empty documents, strings of digits. They will not catch the quality problems that actually matter in web-scraped training data. You need structural signals: repetition, punctuation patterns, boilerplate, and line-level analysis.

**2. The literature is your friend.** We did not invent any of these signals. Rae et al. (Gopher), Raffel et al. (C4), and Penedo et al. (FineWeb) already did the hard work of figuring out which heuristics matter and what thresholds to set. Our contribution is packaging these signals into a single tool that anyone can run with one command.

**3. Verify against code, not just papers.** We cross-referenced every threshold against the [huggingface/datatrove](https://github.com/huggingface/datatrove) source code. Papers round numbers, omit edge cases, and sometimes describe idealized versions of what was actually implemented. The code is the ground truth.

**4. Be honest about your tool's limitations.** Our v1 scorer was useless for its stated purpose, and we shipped it anyway. The right response is not to hide the problem -- it is to measure it, fix it, and show your work. If your quality tool gives an A to everything, say so and improve it.

**5. Domain-specific data is not "low quality."** Code datasets scoring B+ and tweet datasets scoring B are not failures of those datasets. They are reflections of the fact that prose-oriented quality signals do not fully apply to non-prose content. A B+ for clean source code is correct. The scorer is telling you "this is not prose," which is true and useful information.

**6. C4 en.noclean is genuinely messy.** This is not obvious from its v1 score of 94.3%. It becomes obvious when you look at repetition rates (59.4% of documents flagged) and punctuation patterns (65.8% flagged). The gap between C4 en.noclean and C4 en is the gap between raw web crawl and filtered training data. A quality scorer that cannot see this gap is not worth running.

---

## How to Use It

Install Dokime and score your own data:

```bash
pip install dokime
```

Score a local file:

```bash
dokime score your_data.jsonl
```

Score a HuggingFace dataset:

```bash
dokime score --hf allenai/c4 --config en --split validation --sample 1000
```

The output includes a letter grade (A through F), a composite score (0-100), per-signal breakdowns, and the worst-scoring documents for manual inspection. Use it to audit datasets before training, compare curation strategies, or verify that your filtering pipeline is actually working.

The scorer is one piece of Dokime, which also includes 16 heuristic filters, three dedup methods, embedding-based semantic search and outlier detection, and a YAML-configurable pipeline. Everything is open source under Apache 2.0.

**Links:**
- GitHub: [github.com/dokime-ai/dokime](https://github.com/dokime-ai/dokime)
- PyPI: [pypi.org/project/dokime](https://pypi.org/project/dokime/)

---

*The 12 scoring signals in Dokime v2 are derived from Gopher (Rae et al. 2021, [arXiv:2112.11446](https://arxiv.org/abs/2112.11446)), C4/T5 (Raffel et al. 2020, [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)), and FineWeb (Penedo et al. 2024, [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)). All thresholds verified against [huggingface/datatrove](https://github.com/huggingface/datatrove) source code.*
