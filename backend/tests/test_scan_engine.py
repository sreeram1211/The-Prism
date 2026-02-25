"""
Tests for the Prism Scan Engine (Phase 2 mock implementation).

Coverage:
  - MockPrismScanEngine produces deterministic results
  - All 9 dimensions are present and scores are in 0-1 range
  - Family biases shift scores appropriately
  - Geometric separation ratio is in the documented range (125×–1,376×)
  - Dimension subset scanning works
  - get_scan_engine() factory
  - ScanDimension enum
"""

from __future__ import annotations

import pytest

from prism.scan.engine import (
    DimensionScore,
    MockPrismScanEngine,
    PrismScanEngine,
    ScanDimension,
    ScanReport,
    get_scan_engine,
)


MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_ID_2 = "meta-llama/Meta-Llama-3-8B"


class TestScanDimension:
    def test_all_returns_nine_dimensions(self):
        dims = ScanDimension.all()
        assert len(dims) == 9

    def test_all_dimensions_present(self):
        keys = {d.value for d in ScanDimension.all()}
        expected = {
            "sycophancy", "hedging", "calibration", "depth",
            "coherence", "focus", "specificity", "verbosity", "repetition",
        }
        assert keys == expected


class TestMockScanEngine:
    def _engine(self, family: str = "attention") -> MockPrismScanEngine:
        return MockPrismScanEngine(family=family)

    def test_scan_returns_scan_report(self):
        report = self._engine().scan(MODEL_ID)
        assert isinstance(report, ScanReport)

    def test_all_nine_dimensions_returned(self):
        report = self._engine().scan(MODEL_ID)
        returned_dims = {ds.dimension.value for ds in report.scores}
        expected = {d.value for d in ScanDimension.all()}
        assert returned_dims == expected

    def test_scores_are_in_unit_range(self):
        report = self._engine().scan(MODEL_ID)
        for ds in report.scores:
            assert 0.0 <= ds.score <= 1.0, (
                f"{ds.dimension.value} score {ds.score} out of [0,1] range"
            )

    def test_scores_are_deterministic(self):
        """Same model_id must always produce the same scores."""
        r1 = self._engine().scan(MODEL_ID)
        r2 = self._engine().scan(MODEL_ID)
        for a, b in zip(r1.scores, r2.scores):
            assert a.score == b.score
            assert a.interpretation == b.interpretation

    def test_different_models_produce_different_scores(self):
        r1 = self._engine().scan(MODEL_ID)
        r2 = self._engine().scan(MODEL_ID_2)
        scores_a = [ds.score for ds in r1.scores]
        scores_b = [ds.score for ds in r2.scores]
        assert scores_a != scores_b

    def test_geometric_separation_ratio_in_range(self):
        report = self._engine().scan(MODEL_ID)
        assert 125.0 <= report.geometric_separation_ratio <= 1376.0

    def test_scan_duration_is_positive(self):
        report = self._engine().scan(MODEL_ID)
        assert report.scan_duration_ms >= 0.0

    def test_interpretation_strings_are_non_empty(self):
        report = self._engine().scan(MODEL_ID)
        for ds in report.scores:
            assert isinstance(ds.interpretation, str)
            assert len(ds.interpretation) > 0

    def test_dimension_subset_scanning(self):
        dims = [ScanDimension.SYCOPHANCY, ScanDimension.DEPTH]
        report = self._engine().scan(MODEL_ID, dimensions=dims)
        returned = [ds.dimension for ds in report.scores]
        assert returned == dims

    def test_model_id_stored_in_report(self):
        report = self._engine().scan(MODEL_ID)
        assert report.model_id == MODEL_ID

    def test_ssm_family_bias_shifts_verbosity(self):
        """SSM family should score higher verbosity than attention."""
        r_attn = MockPrismScanEngine(family="attention").scan(MODEL_ID)
        r_ssm  = MockPrismScanEngine(family="ssm").scan(MODEL_ID)

        attn_verb = next(ds.score for ds in r_attn.scores if ds.dimension == ScanDimension.VERBOSITY)
        ssm_verb  = next(ds.score for ds in r_ssm.scores  if ds.dimension == ScanDimension.VERBOSITY)

        # SSM bias adds +0.10 to verbosity range, so its score should tend higher
        assert ssm_verb >= attn_verb - 0.05  # allow minor RNG overlap

    def test_encoder_decoder_family_bias(self):
        """Encoder-decoder family should have lower verbosity (−0.08 bias)."""
        r_attn = MockPrismScanEngine(family="attention").scan(MODEL_ID)
        r_encdec = MockPrismScanEngine(family="encoder_decoder").scan(MODEL_ID)

        attn_v = next(ds.score for ds in r_attn.scores  if ds.dimension == ScanDimension.VERBOSITY)
        enc_v  = next(ds.score for ds in r_encdec.scores if ds.dimension == ScanDimension.VERBOSITY)

        assert enc_v <= attn_v + 0.05


class TestGetScanEngine:
    def test_mock_factory_returns_mock_engine(self):
        engine = get_scan_engine(mock=True)
        assert isinstance(engine, MockPrismScanEngine)

    def test_real_engine_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            get_scan_engine(mock=False)

    def test_factory_family_param_passed_through(self):
        engine = get_scan_engine(mock=True, family="ssm")
        assert isinstance(engine, MockPrismScanEngine)
        assert engine.family == "ssm"
