"""Quality scoring — compute per-document quality signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dokime.core.filters import Filter


@dataclass
class TokenCountFilter(Filter):
    """Filter documents by estimated token count.

    Uses a fast heuristic: ~4 characters per token for English text
    (GPT-style tokenization). For exact counts, use a real tokenizer.
    """

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
    """Filter documents by character-level entropy as a perplexity proxy.

    Documents with very low entropy are likely repetitive/synthetic.
    Documents with very high entropy are likely garbage/random.
    Natural language typically falls in a specific entropy range.

    This is a fast CPU-based approximation. For true perplexity scoring,
    use a KenLM model (not included to keep dependencies minimal).
    """

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
    """Compute multiple quality signals for each document without filtering.

    Adds quality scores as new fields to each document, which can then be
    used for sorting, visualization, or downstream filtering.

    Example::

        scorer = QualityScorer()
        for doc in data:
            scored_doc = scorer.score(doc)
            # scored_doc now has: _quality_score, _char_entropy, _estimated_tokens, etc.
    """

    text_field: str = "text"

    def score(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Add quality scores to a document.

        Adds the following fields:
        - _char_count: character count
        - _word_count: word count
        - _estimated_tokens: estimated GPT-style token count
        - _char_entropy: Shannon entropy of character distribution
        - _whitespace_ratio: ratio of whitespace characters
        - _alpha_ratio: ratio of alphabetic characters
        - _special_ratio: ratio of special characters
        - _avg_word_length: average word length
        - _quality_score: composite quality score (0-1, higher = better)
        """
        text = sample.get(self.text_field, "")
        length = len(text)
        words = text.split()

        scored = dict(sample)

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

        # Composite quality score (0-1)
        scored["_quality_score"] = self._composite_score(scored)

        return scored

    @staticmethod
    def _composite_score(scored: dict[str, Any]) -> float:
        """Compute a composite quality score from individual signals."""
        signals = []

        # Entropy: natural text is typically 3.5-5.0 bits/char
        entropy = scored.get("_char_entropy", 0.0)
        if 3.0 <= entropy <= 5.0:
            signals.append(1.0)
        elif 2.0 <= entropy <= 5.5:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # Alpha ratio: natural text is mostly alphabetic
        alpha = scored.get("_alpha_ratio", 0.0)
        signals.append(min(alpha / 0.7, 1.0))

        # Word count: very short or very long docs are suspicious
        words = scored.get("_word_count", 0)
        if 20 <= words <= 10000:
            signals.append(1.0)
        elif 5 <= words <= 50000:
            signals.append(0.5)
        else:
            signals.append(0.0)

        # Average word length: natural text is ~4-6 chars
        avg_wl = scored.get("_avg_word_length", 0.0)
        if 3.0 <= avg_wl <= 8.0:
            signals.append(1.0)
        elif 2.0 <= avg_wl <= 12.0:
            signals.append(0.5)
        else:
            signals.append(0.0)

        return round(sum(signals) / len(signals), 3) if signals else 0.0
