"""Quality scoring — compute per-document quality signals.

Signals are based on verified thresholds from:
- Gopher (Rae et al. 2021, arXiv:2112.11446) — via DataTrove gopher_quality_filter.py
- C4 (Raffel et al. 2020, arXiv:1910.10683) — via DataTrove c4_filters.py
- FineWeb (Penedo et al. 2024, arXiv:2406.17557) — via DataTrove fineweb_quality_filter.py

All thresholds verified against huggingface/datatrove source code on 2026-03-19.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from dokime.core.filters import Filter

# Boilerplate phrases from C4 (verified: DataTrove c4_filters.py)
_C4_BOILERPLATE = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
]

# Terminal punctuation characters (C4 + FineWeb)
_TERMINAL_PUNCT = frozenset('.!?"')

# Sentence-ending pattern
_SENTENCE_END = re.compile(r"[.!?]\s")


@dataclass
class TokenCountFilter(Filter):
    """Filter documents by estimated token count."""

    min_tokens: int = 0
    max_tokens: int = 1_000_000
    chars_per_token: float = 4.0
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        estimated_tokens = len(text) / self.chars_per_token
        return self.min_tokens <= estimated_tokens <= self.max_tokens

    def name(self) -> str:
        return f"TokenCountFilter(min={self.min_tokens}, max={self.max_tokens})"


@dataclass
class PerplexityFilter(Filter):
    """Filter documents by character-level entropy as a perplexity proxy."""

    min_entropy: float = 2.0
    max_entropy: float = 5.5
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        if len(text) < 10:
            return False
        entropy = self._char_entropy(text)
        return self.min_entropy <= entropy <= self.max_entropy

    @staticmethod
    def _char_entropy(text: str) -> float:
        """Compute Shannon entropy of character distribution."""
        if not text:
            return 0.0
        freq: dict[str, int] = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def name(self) -> str:
        return f"PerplexityFilter(min_entropy={self.min_entropy}, max_entropy={self.max_entropy})"


@dataclass
class QualityScorer:
    """Compute per-document quality signals based on Gopher/C4/FineWeb heuristics.

    Signals and thresholds verified against DataTrove source code:
    - gopher_quality_filter.py (word count, avg word length, stop words, symbol ratio)
    - gopher_repetition_filter.py (duplicate lines, top n-grams, duplicate n-grams)
    - c4_filters.py (sentence count, boilerplate, terminal punctuation)
    - fineweb_quality_filter.py (short line ratio, line punctuation ratio)
    """

    text_field: str = "text"

    def score(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Add quality scores to a document."""
        text = sample.get(self.text_field, "")
        length = len(text)
        words = text.split()
        lines = text.split("\n") if text else []
        non_empty_lines = [line for line in lines if line.strip()]

        scored = dict(sample)

        # === Basic signals (existing) ===
        scored["_char_count"] = length
        scored["_word_count"] = len(words)
        scored["_estimated_tokens"] = round(length / 4.0)

        if length > 0:
            scored["_char_entropy"] = round(PerplexityFilter._char_entropy(text), 3)
            scored["_whitespace_ratio"] = round(sum(1 for c in text if c.isspace()) / length, 3)
            scored["_alpha_ratio"] = round(sum(1 for c in text if c.isalpha()) / length, 3)
            scored["_special_ratio"] = round(sum(1 for c in text if not c.isalnum() and not c.isspace()) / length, 3)
        else:
            scored["_char_entropy"] = 0.0
            scored["_whitespace_ratio"] = 0.0
            scored["_alpha_ratio"] = 0.0
            scored["_special_ratio"] = 0.0

        if words:
            scored["_avg_word_length"] = round(sum(len(w) for w in words) / len(words), 1)
        else:
            scored["_avg_word_length"] = 0.0

        # === New signals: Gopher quality (arXiv:2112.11446) ===
        # Verified thresholds from DataTrove gopher_quality_filter.py

        # Stop words presence (Gopher: min_stop_words=2)
        stop_words = {"the", "be", "to", "of", "and", "that", "have", "with"}
        word_lower = [w.lower() for w in words]
        scored["_stop_word_count"] = sum(1 for w in word_lower if w in stop_words)

        # Symbol-to-word ratio (Gopher: max_symbol_word_ratio=0.1 for '#')
        if words:
            hash_count = sum(1 for w in words if "#" in w)
            scored["_symbol_word_ratio"] = round(hash_count / len(words), 3)
        else:
            scored["_symbol_word_ratio"] = 0.0

        # Ellipsis line ratio (Gopher: max_ellipsis_lines_ratio=0.3)
        if non_empty_lines:
            ellipsis_lines = sum(1 for line in non_empty_lines if line.rstrip().endswith(("...", "\u2026")))
            scored["_ellipsis_line_ratio"] = round(ellipsis_lines / len(non_empty_lines), 3)
        else:
            scored["_ellipsis_line_ratio"] = 0.0

        # Bullet line ratio (Gopher: max_bullet_lines_ratio=0.9)
        if non_empty_lines:
            bullet_lines = sum(
                1 for line in non_empty_lines if line.lstrip().startswith(("-", "*", "\u2022", "\u2023"))
            )
            scored["_bullet_line_ratio"] = round(bullet_lines / len(non_empty_lines), 3)
        else:
            scored["_bullet_line_ratio"] = 0.0

        # === New signals: Gopher repetition (arXiv:2112.11446) ===
        # Verified thresholds from DataTrove gopher_repetition_filter.py

        # Duplicate line fraction (Gopher: dup_line_frac=0.3)
        if non_empty_lines:
            line_counts = Counter(line.strip() for line in non_empty_lines)
            dup_lines = sum(count - 1 for count in line_counts.values() if count > 1)
            scored["_dup_line_frac"] = round(dup_lines / len(non_empty_lines), 3)
        else:
            scored["_dup_line_frac"] = 0.0

        # Top 2-gram character fraction (Gopher: top_2_gram <= 0.20)
        scored["_top_2gram_frac"] = self._top_ngram_char_frac(words, 2)
        scored["_top_3gram_frac"] = self._top_ngram_char_frac(words, 3)
        scored["_top_4gram_frac"] = self._top_ngram_char_frac(words, 4)

        # Duplicate 5-gram character fraction (Gopher: dup_5_gram <= 0.15)
        scored["_dup_5gram_frac"] = self._dup_ngram_char_frac(words, 5)

        # === New signals: C4 (arXiv:1910.10683) ===
        # Verified from DataTrove c4_filters.py

        # Sentence count (C4: min 5 sentences — we compute as signal, not hard filter)
        scored["_sentence_count"] = len(_SENTENCE_END.findall(text)) + (
            1 if text.rstrip().endswith((".", "!", "?")) else 0
        )

        # Boilerplate line detection (C4: "terms of use", "cookie policy", etc.)
        text_lower = text.lower()
        scored["_boilerplate_lines"] = sum(1 for phrase in _C4_BOILERPLATE if phrase in text_lower)

        # Curly bracket presence (C4: reject documents with '{')
        scored["_has_curly_bracket"] = 1 if "{" in text else 0

        # === New signals: FineWeb (arXiv:2406.17557) ===
        # Verified from DataTrove fineweb_quality_filter.py

        # Line punctuation ratio (FineWeb: line_punct_thr=0.12)
        if non_empty_lines:
            punct_lines = sum(1 for line in non_empty_lines if line.rstrip()[-1:] in _TERMINAL_PUNCT)
            scored["_line_punct_ratio"] = round(punct_lines / len(non_empty_lines), 3)
        else:
            scored["_line_punct_ratio"] = 0.0

        # Short line ratio (FineWeb: short_line_thr=0.67, short_line_length=30)
        if non_empty_lines:
            short_lines = sum(1 for line in non_empty_lines if len(line.strip()) <= 30)
            scored["_short_line_ratio"] = round(short_lines / len(non_empty_lines), 3)
        else:
            scored["_short_line_ratio"] = 0.0

        # List ratio — newlines to words (FineWeb: new_line_ratio=0.3)
        if words:
            scored["_list_ratio"] = round(len(lines) / len(words), 3)
        else:
            scored["_list_ratio"] = 0.0

        # Composite quality score
        scored["_quality_score"] = self._composite_score(scored)

        return scored

    @staticmethod
    def _top_ngram_char_frac(words: list[str], n: int) -> float:
        """Fraction of chars in document belonging to the most common n-gram.

        Source: Gopher repetition filter (DataTrove gopher_repetition_filter.py).
        """
        if len(words) < n:
            return 0.0
        ngrams: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i : i + n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        if not ngrams:
            return 0.0
        top_gram, top_count = max(ngrams.items(), key=lambda x: x[1])
        if top_count <= 1:
            return 0.0
        gram_chars = sum(len(w) for w in top_gram) + n - 1  # words + spaces
        total_chars = sum(len(w) for w in words) + len(words) - 1
        if total_chars == 0:
            return 0.0
        return round((gram_chars * top_count) / total_chars, 3)

    @staticmethod
    def _dup_ngram_char_frac(words: list[str], n: int) -> float:
        """Fraction of chars in document belonging to ANY duplicated n-gram.

        Source: Gopher repetition filter (DataTrove gopher_repetition_filter.py).
        """
        if len(words) < n:
            return 0.0
        ngrams: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i : i + n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        total_chars = sum(len(w) for w in words) + len(words) - 1
        if total_chars == 0:
            return 0.0
        dup_chars = 0
        for gram, count in ngrams.items():
            if count > 1:
                gram_chars = sum(len(w) for w in gram) + n - 1
                dup_chars += gram_chars * count
        return round(min(dup_chars / total_chars, 1.0), 3)

    @staticmethod
    def _composite_score(scored: dict[str, Any]) -> float:
        """Composite quality score from 12 signals (0-1, higher = better).

        Thresholds based on:
        - Gopher (arXiv:2112.11446): word count, avg word length, stop words,
          duplicate lines, top n-grams, duplicate n-grams
        - C4 (arXiv:1910.10683): sentence count, boilerplate
        - FineWeb (arXiv:2406.17557): line punctuation, short lines
        """
        # Empty or near-empty documents always score 0
        if scored.get("_word_count", 0) < 5:
            return 0.0

        signals = []

        # 1. Entropy (existing — natural text: 3.0-5.0 bits/char)
        entropy = scored.get("_char_entropy", 0.0)
        if 3.0 <= entropy <= 5.0:
            signals.append(1.0)
        elif 2.0 <= entropy <= 5.5:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 2. Alpha ratio (existing)
        alpha = scored.get("_alpha_ratio", 0.0)
        signals.append(min(alpha / 0.7, 1.0))

        # 3. Word count (Gopher: 50-100K)
        words = scored.get("_word_count", 0)
        if 50 <= words <= 100_000:
            signals.append(1.0)
        elif 20 <= words <= 100_000:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 4. Avg word length (Gopher: 3-10)
        avg_wl = scored.get("_avg_word_length", 0.0)
        if 3.0 <= avg_wl <= 10.0:
            signals.append(1.0)
        elif 2.0 <= avg_wl <= 12.0:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 5. Stop words (Gopher: min 2)
        stop_count = scored.get("_stop_word_count", 0)
        signals.append(1.0 if stop_count >= 2 else 0.0)

        # 6. Line punctuation ratio (FineWeb: >= 0.12)
        line_punct = scored.get("_line_punct_ratio", 0.0)
        if line_punct >= 0.12:
            signals.append(1.0)
        elif line_punct >= 0.06:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 7. Short line ratio (FineWeb: <= 0.67)
        short_line = scored.get("_short_line_ratio", 0.0)
        if short_line <= 0.67:
            signals.append(1.0)
        elif short_line <= 0.85:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 8. Duplicate line fraction (Gopher: <= 0.30)
        dup_line = scored.get("_dup_line_frac", 0.0)
        if dup_line <= 0.30:
            signals.append(1.0)
        elif dup_line <= 0.50:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 9. Top 2-gram fraction (Gopher: <= 0.20)
        top2 = scored.get("_top_2gram_frac", 0.0)
        if top2 <= 0.20:
            signals.append(1.0)
        elif top2 <= 0.35:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 10. Duplicate 5-gram fraction (Gopher: <= 0.15)
        dup5 = scored.get("_dup_5gram_frac", 0.0)
        if dup5 <= 0.15:
            signals.append(1.0)
        elif dup5 <= 0.30:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 11. Sentence count (C4: >= 5)
        sent = scored.get("_sentence_count", 0)
        if sent >= 5:
            signals.append(1.0)
        elif sent >= 2:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # 12. Boilerplate (C4: 0 boilerplate phrases)
        boilerplate = scored.get("_boilerplate_lines", 0)
        signals.append(1.0 if boilerplate == 0 else 0.0)

        return round(sum(signals) / len(signals), 3) if signals else 0.0
