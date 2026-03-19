# Filters

All filters inherit from `dokime.core.filters.Filter`. They return `True` to keep a sample, `False` to discard it. Every filter accepts a `text_field` parameter (default: `"text"`).

## Length and Structure

### LengthFilter

Filter by character count.

```python
from dokime.core.filters import LengthFilter
f = LengthFilter(min_length=50, max_length=100_000)
```

### WordCountFilter

Filter by word count.

```python
from dokime.core.filters import WordCountFilter
f = WordCountFilter(min_words=10, max_words=50_000)
```

### LineCountFilter

Filter by number of lines.

```python
from dokime.core.filters import LineCountFilter
f = LineCountFilter(min_lines=1, max_lines=10_000)
```

## Content Quality

### WhitespaceFilter

Remove documents with excessive whitespace (formatting junk, empty pages).

```python
from dokime.core.filters import WhitespaceFilter
f = WhitespaceFilter(max_whitespace_ratio=0.4)
```

### RepetitionFilter

Catch boilerplate and spam via n-gram repetition ratio.

```python
from dokime.core.filters import RepetitionFilter
f = RepetitionFilter(max_repetition_ratio=0.3, ngram_size=5)
```

### SpecialCharFilter

Remove documents overloaded with special characters (encoding artifacts, garbled text).

```python
from dokime.core.filters import SpecialCharFilter
f = SpecialCharFilter(max_special_ratio=0.3)
```

### AlphaFilter

Catch numeric spam, base64 blobs, and other non-text content.

```python
from dokime.core.filters import AlphaFilter
f = AlphaFilter(min_alpha_ratio=0.5)
```

## URL and Boilerplate

### URLFilter

Remove URL-heavy documents (link farms, crawl artifacts).

```python
from dokime.core.filters import URLFilter
f = URLFilter(max_url_ratio=0.1)
```

### StopwordFilter

Catch keyword spam, code, and list-only content by requiring a minimum stopword presence.

```python
from dokime.core.filters import StopwordFilter
f = StopwordFilter(min_stopword_ratio=0.05)
```

## Language Detection

### LanguageFilter

Filter by detected language using lingua (or fastText fallback). Requires `dokime[nlp]`.

```python
from dokime.core.filters import LanguageFilter
f = LanguageFilter(languages=["en", "de"], min_confidence=0.5)
```

## Field and Pattern

### FieldExistsFilter

Ensure a required field is present and non-empty.

```python
from dokime.core.filters import FieldExistsFilter
f = FieldExistsFilter(required_field="text")
```

### RegexFilter

Include or exclude documents matching a regex pattern.

```python
from dokime.core.filters import RegexFilter

# Exclude boilerplate
f = RegexFilter(pattern=r"(?i)cookie policy|terms of service", exclude=True)

# Keep only documents containing a keyword
f = RegexFilter(pattern=r"machine learning", exclude=False)
```

## Deduplication Filters

### ExactDedup

SHA-256 exact deduplication. No extra dependencies.

```python
from dokime.quality.dedup import ExactDedup
f = ExactDedup()
```

### MinHashDedup

Fuzzy deduplication via MinHash-LSH. Requires `dokime[dedup]`.

```python
from dokime.quality.dedup import MinHashDedup
f = MinHashDedup(threshold=0.8, num_perm=128)
```
