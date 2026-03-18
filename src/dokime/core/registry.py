"""Filter registry for YAML config-based pipeline construction."""

from __future__ import annotations

from dokime.core.filters import (
    AlphaFilter,
    FieldExistsFilter,
    Filter,
    LengthFilter,
    LineCountFilter,
    RegexFilter,
    RepetitionFilter,
    SpecialCharFilter,
    StopwordFilter,
    URLFilter,
    WhitespaceFilter,
    WordCountFilter,
)

# Registry mapping filter names to classes.
# Dedup filters are registered lazily to avoid import errors when optional deps aren't installed.
FILTER_REGISTRY: dict[str, type[Filter]] = {
    "LengthFilter": LengthFilter,
    "WordCountFilter": WordCountFilter,
    "LineCountFilter": LineCountFilter,
    "WhitespaceFilter": WhitespaceFilter,
    "RepetitionFilter": RepetitionFilter,
    "SpecialCharFilter": SpecialCharFilter,
    "AlphaFilter": AlphaFilter,
    "URLFilter": URLFilter,
    "StopwordFilter": StopwordFilter,
    "FieldExistsFilter": FieldExistsFilter,
    "RegexFilter": RegexFilter,
}


def _register_optional_filters() -> None:
    """Register filters that require optional dependencies."""
    try:
        from dokime.quality.dedup import ExactDedup, MinHashDedup

        FILTER_REGISTRY["ExactDedup"] = ExactDedup
        FILTER_REGISTRY["MinHashDedup"] = MinHashDedup
    except ImportError:
        pass

    try:
        from dokime.core.filters import LanguageFilter

        FILTER_REGISTRY["LanguageFilter"] = LanguageFilter
    except ImportError:
        pass

    # Quality scoring filters (no extra deps needed)
    from dokime.quality.scoring import PerplexityFilter, TokenCountFilter

    FILTER_REGISTRY["TokenCountFilter"] = TokenCountFilter
    FILTER_REGISTRY["PerplexityFilter"] = PerplexityFilter


_register_optional_filters()


def register_filter(name: str, filter_cls: type[Filter]) -> None:
    """Register a custom filter class for use in YAML configs."""
    FILTER_REGISTRY[name] = filter_cls
