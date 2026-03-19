"""Tests for spectral analysis of vision encoder Jacobians."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dokime.spectral import SpectralAnalyzer, SpectralComparison
from dokime.spectral.analyzer import _auroc_from_scores, _cohens_d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyEncoder(nn.Module):
    """Minimal differentiable encoder: (B, 3, 8, 8) -> (B, 4)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return TinyEncoder()


@pytest.fixture
def analyzer(encoder):
    return SpectralAnalyzer(encoder, device="cpu")


@pytest.fixture
def dummy_images():
    """4 random images of shape (4, 3, 8, 8)."""
    torch.manual_seed(1)
    return torch.randn(4, 3, 8, 8)


# ---------------------------------------------------------------------------
# SpectralAnalyzer.compute_spectra
# ---------------------------------------------------------------------------


class TestComputeSpectra:
    def test_returns_correct_shape(self, analyzer, dummy_images):
        spectra = analyzer.compute_spectra(dummy_images[:2])
        assert spectra.shape == (2, 4)  # 2 images, embedding dim 4

    def test_singular_values_nonnegative(self, analyzer, dummy_images):
        spectra = analyzer.compute_spectra(dummy_images[:2])
        assert np.all(spectra >= 0)

    def test_singular_values_descending(self, analyzer, dummy_images):
        spectra = analyzer.compute_spectra(dummy_images[:2])
        for row in spectra:
            assert np.all(np.diff(row) <= 1e-6), "SVs should be in descending order"

    def test_randomized_method(self, analyzer, dummy_images):
        spectra = analyzer.compute_spectra(
            dummy_images[:2], method="randomized", n_proj=3,
        )
        # n_proj < d=4, so shape is (2, 3)
        assert spectra.shape == (2, 3)
        assert np.all(spectra >= 0)

    def test_invalid_method_raises(self, analyzer, dummy_images):
        with pytest.raises(ValueError, match="Unknown method"):
            analyzer.compute_spectra(dummy_images[:1], method="bogus")

    def test_verbose_does_not_crash(self, analyzer, dummy_images, capsys):
        analyzer.compute_spectra(dummy_images[:1], verbose=True)
        captured = capsys.readouterr()
        assert "1/1" in captured.out

    def test_deterministic_full(self, analyzer, dummy_images):
        s1 = analyzer.compute_spectra(dummy_images[:1])
        s2 = analyzer.compute_spectra(dummy_images[:1])
        np.testing.assert_allclose(s1, s2, atol=1e-5)

    def test_deterministic_randomized(self, analyzer, dummy_images):
        s1 = analyzer.compute_spectra(dummy_images[:1], method="randomized", seed=99)
        s2 = analyzer.compute_spectra(dummy_images[:1], method="randomized", seed=99)
        np.testing.assert_allclose(s1, s2, atol=1e-5)


# ---------------------------------------------------------------------------
# SpectralAnalyzer.compare_distributions
# ---------------------------------------------------------------------------


class TestCompareDistributions:
    def test_returns_spectral_comparison(self, analyzer, dummy_images):
        id_spec = analyzer.compute_spectra(dummy_images[:2])
        ood_spec = analyzer.compute_spectra(dummy_images[2:])
        result = analyzer.compare_distributions(id_spec, ood_spec)
        assert isinstance(result, SpectralComparison)

    def test_auroc_range(self, analyzer, dummy_images):
        id_spec = analyzer.compute_spectra(dummy_images[:2])
        ood_spec = analyzer.compute_spectra(dummy_images[2:])
        result = analyzer.compare_distributions(id_spec, ood_spec)
        assert 0.0 <= result.auroc_standard <= 1.0
        assert 0.0 <= result.auroc_tail_weighted <= 1.0

    def test_top_k_correlation_range(self, analyzer, dummy_images):
        id_spec = analyzer.compute_spectra(dummy_images[:2])
        ood_spec = analyzer.compute_spectra(dummy_images[2:])
        result = analyzer.compare_distributions(id_spec, ood_spec, top_k=2)
        assert -1.0 <= result.top_k_correlation <= 1.0

    def test_per_band_keys(self, analyzer, dummy_images):
        id_spec = analyzer.compute_spectra(dummy_images[:2])
        ood_spec = analyzer.compute_spectra(dummy_images[2:])
        result = analyzer.compare_distributions(id_spec, ood_spec)
        expected_bands = {"top_quartile", "upper_mid", "lower_mid", "tail_quartile"}
        assert set(result.per_band_discrimination.keys()) == expected_bands

    def test_identical_distributions_low_discrimination(self, analyzer, dummy_images):
        """If ID and OOD are the same data, Cohen's d should be ~0."""
        spectra = analyzer.compute_spectra(dummy_images)
        result = analyzer.compare_distributions(spectra, spectra)
        assert abs(result.cohens_d) < 0.01

    def test_separated_distributions_high_discrimination(self, analyzer):
        """Synthetic spectra with clear separation should yield high Cohen's d."""
        rng = np.random.RandomState(42)
        id_spectra = rng.exponential(scale=10.0, size=(50, 4))
        id_spectra.sort(axis=1)
        id_spectra = id_spectra[:, ::-1].copy()  # descending

        ood_spectra = rng.exponential(scale=0.1, size=(50, 4))
        ood_spectra.sort(axis=1)
        ood_spectra = ood_spectra[:, ::-1].copy()

        result = analyzer.compare_distributions(id_spectra, ood_spectra)
        assert abs(result.cohens_d) > 1.0
        assert result.auroc_standard > 0.9 or result.auroc_standard < 0.1

    def test_optimal_drop_k_nonnegative(self, analyzer, dummy_images):
        id_spec = analyzer.compute_spectra(dummy_images[:2])
        ood_spec = analyzer.compute_spectra(dummy_images[2:])
        result = analyzer.compare_distributions(id_spec, ood_spec)
        assert result.optimal_drop_k >= 0


# ---------------------------------------------------------------------------
# SpectralAnalyzer.report
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_returns_comparison(self, analyzer, dummy_images):
        id_spec = analyzer.compute_spectra(dummy_images[:2])
        ood_spec = analyzer.compute_spectra(dummy_images[2:])
        result = analyzer.report(id_spec, ood_spec)
        assert isinstance(result, SpectralComparison)

    def test_report_prints_output(self, analyzer, dummy_images, capsys):
        id_spec = analyzer.compute_spectra(dummy_images[:2])
        ood_spec = analyzer.compute_spectra(dummy_images[2:])
        analyzer.report(id_spec, ood_spec)
        captured = capsys.readouterr()
        # Should contain some report text (either rich or plain)
        assert len(captured.out) > 50


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_cohens_d_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(_cohens_d(a, a)) < 1e-10

    def test_cohens_d_separated(self):
        a = np.array([10.0, 11.0, 12.0, 13.0])
        b = np.array([0.0, 1.0, 2.0, 3.0])
        d = _cohens_d(a, b)
        assert d > 5.0  # very well separated

    def test_cohens_d_small_sample(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert _cohens_d(a, b) == 0.0  # requires n >= 2 each

    def test_auroc_perfect(self):
        id_scores = np.array([10.0, 11.0, 12.0])
        ood_scores = np.array([1.0, 2.0, 3.0])
        assert _auroc_from_scores(id_scores, ood_scores) == 1.0

    def test_auroc_random(self):
        rng = np.random.RandomState(0)
        scores = rng.randn(200)
        auroc = _auroc_from_scores(scores[:100], scores[100:])
        assert 0.3 < auroc < 0.7  # near chance

    def test_auroc_inverted(self):
        id_scores = np.array([1.0, 2.0, 3.0])
        ood_scores = np.array([10.0, 11.0, 12.0])
        assert _auroc_from_scores(id_scores, ood_scores) == 0.0
