"""Tests for core heuristic filters."""

from dokime.core.filters import (
    AlphaFilter,
    FieldExistsFilter,
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


class TestLengthFilter:
    def test_keeps_normal_text(self):
        f = LengthFilter(min_length=10, max_length=1000)
        assert f.filter({"text": "This is a normal length document."})

    def test_rejects_too_short(self):
        f = LengthFilter(min_length=100)
        assert not f.filter({"text": "Short."})

    def test_rejects_too_long(self):
        f = LengthFilter(max_length=10)
        assert not f.filter({"text": "This is way too long for the filter."})

    def test_empty_text(self):
        f = LengthFilter(min_length=1)
        assert not f.filter({"text": ""})


class TestWordCountFilter:
    def test_keeps_normal(self):
        f = WordCountFilter(min_words=3, max_words=100)
        assert f.filter({"text": "This has enough words to pass."})

    def test_rejects_too_few(self):
        f = WordCountFilter(min_words=10)
        assert not f.filter({"text": "Too few."})

    def test_rejects_too_many(self):
        f = WordCountFilter(max_words=3)
        assert not f.filter({"text": "This has way more than three words in it."})


class TestLineCountFilter:
    def test_keeps_normal(self):
        f = LineCountFilter(min_lines=1, max_lines=10)
        assert f.filter({"text": "Line one\nLine two\nLine three"})

    def test_rejects_too_few(self):
        f = LineCountFilter(min_lines=5)
        assert not f.filter({"text": "Just one line"})


class TestWhitespaceFilter:
    def test_keeps_normal_text(self):
        f = WhitespaceFilter(max_whitespace_ratio=0.5)
        assert f.filter({"text": "This is normal text with spaces."})

    def test_rejects_excessive_whitespace(self):
        f = WhitespaceFilter(max_whitespace_ratio=0.3)
        assert not f.filter({"text": "   \n\n\n   \t\t   a"})

    def test_rejects_empty(self):
        f = WhitespaceFilter()
        assert not f.filter({"text": ""})


class TestRepetitionFilter:
    def test_keeps_varied_text(self):
        f = RepetitionFilter(max_repetition_ratio=0.5)
        assert f.filter({"text": "The quick brown fox jumps over the lazy dog and runs away fast"})

    def test_rejects_repetitive_text(self):
        f = RepetitionFilter(max_repetition_ratio=0.3, ngram_size=3)
        repeated = " ".join(["the quick brown"] * 50)
        assert not f.filter({"text": repeated})

    def test_short_text_passes(self):
        f = RepetitionFilter(ngram_size=5)
        assert f.filter({"text": "Hi there"})


class TestSpecialCharFilter:
    def test_keeps_normal_text(self):
        f = SpecialCharFilter(max_special_ratio=0.3)
        assert f.filter({"text": "Normal text with some punctuation."})

    def test_rejects_garbage(self):
        f = SpecialCharFilter(max_special_ratio=0.2)
        assert not f.filter({"text": "###@@@!!!***&&&^^^%%%$$$"})


class TestAlphaFilter:
    def test_keeps_normal_text(self):
        f = AlphaFilter(min_alpha_ratio=0.5)
        assert f.filter({"text": "This is mostly alphabetic text."})

    def test_rejects_numeric_spam(self):
        f = AlphaFilter(min_alpha_ratio=0.5)
        assert not f.filter({"text": "123456789 0000 11111 22222 33333"})

    def test_rejects_empty(self):
        f = AlphaFilter()
        assert not f.filter({"text": ""})


class TestURLFilter:
    def test_keeps_normal_text(self):
        f = URLFilter(max_url_ratio=0.3)
        assert f.filter({"text": "Check out this article about machine learning."})

    def test_rejects_url_heavy(self):
        f = URLFilter(max_url_ratio=0.1)
        assert not f.filter({"text": "Visit https://example.com/very/long/path/to/something"})


class TestStopwordFilter:
    def test_keeps_natural_text(self):
        f = StopwordFilter(min_stopword_ratio=0.1)
        assert f.filter({"text": "This is a normal sentence with many common words in it."})

    def test_rejects_keyword_spam(self):
        f = StopwordFilter(min_stopword_ratio=0.1)
        assert not f.filter({"text": "bitcoin crypto blockchain NFT defi yield staking mining"})

    def test_rejects_empty(self):
        f = StopwordFilter()
        assert not f.filter({"text": ""})


class TestFieldExistsFilter:
    def test_keeps_if_field_present(self):
        f = FieldExistsFilter(required_field="text")
        assert f.filter({"text": "hello", "id": 1})

    def test_rejects_if_field_missing(self):
        f = FieldExistsFilter(required_field="text")
        assert not f.filter({"id": 1})

    def test_rejects_if_field_empty(self):
        f = FieldExistsFilter(required_field="text")
        assert not f.filter({"text": "", "id": 1})


class TestRegexFilter:
    def test_excludes_matching(self):
        f = RegexFilter(pattern=r"(?i)buy now|click here|subscribe", exclude=True)
        assert not f.filter({"text": "Click here to buy now!"})
        assert f.filter({"text": "This is a normal educational document."})

    def test_includes_matching(self):
        f = RegexFilter(pattern=r"\b(def|class|import)\b", exclude=False)
        assert f.filter({"text": "def hello_world(): pass"})
        assert not f.filter({"text": "No code here."})
