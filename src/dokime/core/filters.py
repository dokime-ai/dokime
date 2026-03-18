"""Base filter classes and built-in heuristic filters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class Filter(ABC):
    """Base class for all filters in a curation pipeline."""

    @abstractmethod
    def filter(self, sample: dict[str, Any]) -> bool:
        """Return True to keep the sample, False to discard it."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this filter."""
        ...


# ---------------------------------------------------------------------------
# Length and structure filters
# ---------------------------------------------------------------------------


@dataclass
class LengthFilter(Filter):
    """Filter documents by text length (character count)."""

    min_length: int = 0
    max_length: int = 1_000_000
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        return self.min_length <= len(text) <= self.max_length

    def name(self) -> str:
        return f"LengthFilter(min={self.min_length}, max={self.max_length})"


@dataclass
class WordCountFilter(Filter):
    """Filter documents by word count."""

    min_words: int = 0
    max_words: int = 1_000_000
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        count = len(text.split())
        return self.min_words <= count <= self.max_words

    def name(self) -> str:
        return f"WordCountFilter(min={self.min_words}, max={self.max_words})"


@dataclass
class LineCountFilter(Filter):
    """Filter documents by number of lines."""

    min_lines: int = 0
    max_lines: int = 1_000_000
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        count = text.count("\n") + 1 if text else 0
        return self.min_lines <= count <= self.max_lines

    def name(self) -> str:
        return f"LineCountFilter(min={self.min_lines}, max={self.max_lines})"


# ---------------------------------------------------------------------------
# Content quality filters
# ---------------------------------------------------------------------------


@dataclass
class WhitespaceFilter(Filter):
    """Filter documents with excessive whitespace ratio."""

    max_whitespace_ratio: float = 0.5
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        if not text:
            return False
        whitespace_count = sum(1 for c in text if c.isspace())
        return (whitespace_count / len(text)) <= self.max_whitespace_ratio

    def name(self) -> str:
        return f"WhitespaceFilter(max_ratio={self.max_whitespace_ratio})"


@dataclass
class RepetitionFilter(Filter):
    """Filter documents with excessive n-gram repetition."""

    max_repetition_ratio: float = 0.3
    ngram_size: int = 5
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        words = text.split()
        if len(words) < self.ngram_size:
            return True

        ngrams = [tuple(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]
        if not ngrams:
            return True

        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
        return repetition_ratio <= self.max_repetition_ratio

    def name(self) -> str:
        return f"RepetitionFilter(max_ratio={self.max_repetition_ratio}, n={self.ngram_size})"


@dataclass
class SpecialCharFilter(Filter):
    """Filter documents with excessive special characters."""

    max_special_ratio: float = 0.3
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        if not text:
            return False
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return (special_count / len(text)) <= self.max_special_ratio

    def name(self) -> str:
        return f"SpecialCharFilter(max_ratio={self.max_special_ratio})"


@dataclass
class AlphaFilter(Filter):
    """Filter documents with too few alphabetic characters (catches garbage/numeric spam)."""

    min_alpha_ratio: float = 0.5
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        if not text:
            return False
        alpha_count = sum(1 for c in text if c.isalpha())
        return (alpha_count / len(text)) >= self.min_alpha_ratio

    def name(self) -> str:
        return f"AlphaFilter(min_ratio={self.min_alpha_ratio})"


# ---------------------------------------------------------------------------
# URL / boilerplate filters
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


@dataclass
class URLFilter(Filter):
    """Filter documents with excessive URLs."""

    max_url_ratio: float = 0.1
    text_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        if not text:
            return False
        url_chars = sum(len(m.group()) for m in _URL_PATTERN.finditer(text))
        return (url_chars / len(text)) <= self.max_url_ratio

    def name(self) -> str:
        return f"URLFilter(max_ratio={self.max_url_ratio})"


@dataclass
class StopwordFilter(Filter):
    """Filter documents with too few stopwords (catches keyword spam, lists, code)."""

    min_stopword_ratio: float = 0.05
    text_field: str = "text"
    stopwords: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "shall",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
                "at",
                "by",
                "from",
                "as",
                "into",
                "about",
                "than",
                "after",
                "before",
                "between",
                "under",
                "above",
                "and",
                "but",
                "or",
                "not",
                "no",
                "if",
                "then",
                "so",
                "that",
                "this",
                "it",
                "its",
                "he",
                "she",
                "they",
                "we",
                "you",
                "i",
                "my",
                "your",
                "his",
                "her",
                "their",
                "our",
            }
        )
    )

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        words = text.lower().split()
        if not words:
            return False
        stopword_count = sum(1 for w in words if w in self.stopwords)
        return (stopword_count / len(words)) >= self.min_stopword_ratio

    def name(self) -> str:
        return f"StopwordFilter(min_ratio={self.min_stopword_ratio})"


# ---------------------------------------------------------------------------
# Language detection filter
# ---------------------------------------------------------------------------


@dataclass
class LanguageFilter(Filter):
    """Filter documents by detected language.

    Uses lingua-language-detector (pure Python, no compilation needed).
    Falls back to fastText if installed and lingua is not available.

    Requires: pip install dokime-ai[nlp]

    Language codes: ISO 639-1 lowercase (e.g., "en", "de", "fr", "zh", "ja").
    """

    languages: list[str] = field(default_factory=lambda: ["en"])
    min_confidence: float = 0.5
    text_field: str = "text"
    _detector: Any = field(default=None, repr=False, init=False)
    _backend: str = field(default="", repr=False, init=False)

    def __post_init__(self) -> None:
        # Try lingua first (pure Python, works everywhere)
        try:
            from lingua import LanguageDetectorBuilder  # noqa: F401

            self._backend = "lingua"
            return
        except ImportError:
            pass

        # Fall back to fasttext
        try:
            import fasttext  # noqa: F401

            self._backend = "fasttext"
            return
        except ImportError:
            pass

        raise ImportError(
            "Install language detection: pip install dokime-ai[nlp]\n  (installs lingua-language-detector)"
        )

    def _get_detector(self) -> Any:
        if self._detector is not None:
            return self._detector

        if self._backend == "lingua":
            from lingua import Language, LanguageDetectorBuilder

            # Map ISO 639-1 codes to lingua Language enums
            lang_map = {lang.iso_code_639_1.name.lower(): lang for lang in Language.all()}
            target_langs = [lang_map[code] for code in self.languages if code in lang_map]

            if not target_langs:
                self._detector = (
                    LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0.1).build()
                )
            else:
                # Build detector for all languages (needed for accurate detection)
                # but we'll filter results to target languages in filter()
                self._detector = LanguageDetectorBuilder.from_all_languages().build()

        elif self._backend == "fasttext":
            import urllib.request
            from pathlib import Path

            import fasttext

            fasttext.FastText.eprint = lambda x: None
            model_path = Path.home() / ".cache" / "dokime" / "lid.176.ftz"
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
                urllib.request.urlretrieve(url, model_path)
            self._detector = fasttext.load_model(str(model_path))

        return self._detector

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        if not text:
            return False

        clean_text = text.replace("\n", " ").strip()[:1000]
        if not clean_text:
            return False

        detector = self._get_detector()

        if self._backend == "lingua":
            result = detector.detect_language_of(clean_text)
            if result is None:
                return False
            # Get ISO 639-1 code
            iso = result.iso_code_639_1
            if iso is None:
                return False
            detected_lang = iso.name.lower()

            # Confidence via compute_language_confidence
            confidence = detector.compute_language_confidence(clean_text, result)
            return detected_lang in self.languages and confidence >= self.min_confidence

        elif self._backend == "fasttext":
            predictions = detector.predict(clean_text)
            label = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]
            return label in self.languages and confidence >= self.min_confidence

        return False

    def name(self) -> str:
        langs = ",".join(self.languages)
        return f"LanguageFilter(langs=[{langs}], min_conf={self.min_confidence}, backend={self._backend})"


# ---------------------------------------------------------------------------
# Field-based filters
# ---------------------------------------------------------------------------


@dataclass
class FieldExistsFilter(Filter):
    """Filter documents that are missing a required field."""

    required_field: str = "text"

    def filter(self, sample: dict[str, Any]) -> bool:
        value = sample.get(self.required_field)
        return value is not None and value != ""

    def name(self) -> str:
        return f"FieldExistsFilter(field={self.required_field})"


@dataclass
class RegexFilter(Filter):
    """Filter documents matching (or not matching) a regex pattern."""

    pattern: str = ""
    exclude: bool = True
    text_field: str = "text"
    _compiled: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern)

    def filter(self, sample: dict[str, Any]) -> bool:
        text = sample.get(self.text_field, "")
        match = bool(self._compiled.search(text))
        return not match if self.exclude else match

    def name(self) -> str:
        mode = "exclude" if self.exclude else "include"
        return f"RegexFilter({mode}, pattern={self.pattern!r})"
