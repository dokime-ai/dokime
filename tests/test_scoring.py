"""Tests for quality scoring and new filters."""

from dokime.quality.scoring import PerplexityFilter, QualityScorer, TokenCountFilter


class TestTokenCountFilter:
    def test_keeps_normal(self):
        f = TokenCountFilter(min_tokens=5, max_tokens=1000)
        assert f.filter({"text": "This is a normal sentence with enough tokens to pass the filter."})

    def test_rejects_too_few(self):
        f = TokenCountFilter(min_tokens=100)
        assert not f.filter({"text": "Short."})

    def test_rejects_too_many(self):
        f = TokenCountFilter(max_tokens=5)
        assert not f.filter({"text": "This sentence has way more than five estimated tokens in it."})


class TestPerplexityFilter:
    def test_keeps_natural_text(self):
        f = PerplexityFilter(min_entropy=2.0, max_entropy=5.5)
        assert f.filter(
            {"text": "Machine learning is a fascinating field of study that combines statistics and computing."}
        )

    def test_rejects_repetitive(self):
        f = PerplexityFilter(min_entropy=2.0, max_entropy=5.5)
        assert not f.filter({"text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})

    def test_rejects_short(self):
        f = PerplexityFilter()
        assert not f.filter({"text": "Hi"})


class TestQualityScorer:
    def test_adds_scores(self):
        scorer = QualityScorer()
        doc = {"text": "This is a normal English sentence about data quality.", "id": 1}
        scored = scorer.score(doc)

        assert "_char_count" in scored
        assert "_word_count" in scored
        assert "_estimated_tokens" in scored
        assert "_char_entropy" in scored
        assert "_whitespace_ratio" in scored
        assert "_alpha_ratio" in scored
        assert "_quality_score" in scored
        assert scored["id"] == 1  # original fields preserved

    def test_quality_score_range(self):
        scorer = QualityScorer()
        doc = {"text": "This is a perfectly normal document with good quality content about machine learning."}
        scored = scorer.score(doc)

        assert 0.0 <= scored["_quality_score"] <= 1.0

    def test_empty_text(self):
        scorer = QualityScorer()
        scored = scorer.score({"text": ""})
        assert scored["_quality_score"] == 0.0

    def test_garbage_gets_low_score(self):
        scorer = QualityScorer()
        good = scorer.score({"text": "This is a well-written document about the importance of data quality."})
        bad = scorer.score({"text": "aaaa bbbb cccc dddd eeee ffff aaaa bbbb cccc dddd eeee ffff"})
        assert good["_quality_score"] > bad["_quality_score"]
