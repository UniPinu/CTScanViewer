"""The risk layer must refuse to invent numbers, and must reject impossible ones.

The calibration path is exercised with synthetic labels, because the real ones
need the CDAS application (CONTEXT.md §1.2) — but the machinery has to be proven
working before those labels land, not after.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.risk import (
    HORIZONS,
    Calibrator,
    RiskPrediction,
    RiskUnavailable,
    UnavailableBackend,
    cohort_percentile,
)


class TestUnavailableBackend:
    def test_raises_rather_than_returning_a_default(self):
        backend = UnavailableBackend()
        assert not backend.available()
        with pytest.raises(RiskUnavailable):
            backend.predict(None)

    def test_describe_reports_not_ok(self):
        assert UnavailableBackend(reason="nope").describe() == {"ok": False, "reason": "nope"}


class TestValidation:
    @staticmethod
    def prediction(scores):
        return RiskPrediction(scores=list(scores), model="test")

    def test_a_plausible_curve_validates(self):
        self.prediction([0.01, 0.02, 0.04, 0.06, 0.09, 0.12]).validate()

    def test_a_flat_curve_validates(self):
        self.prediction([0.05] * 6).validate()

    def test_probabilities_above_one_are_rejected(self):
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            self.prediction([0.1, 0.2, 0.3, 0.4, 0.5, 1.4]).validate()

    def test_negative_probabilities_are_rejected(self):
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            self.prediction([-0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).validate()

    def test_decreasing_cumulative_risk_is_rejected(self):
        """Cumulative risk cannot fall: the year-6 window contains the year-1 one.

        A decreasing curve means the output was misread — a transposed array, or
        the wrong tensor — and rendering it would be worse than failing.
        """
        with pytest.raises(ValueError, match="decreases"):
            self.prediction([0.10, 0.09, 0.12, 0.13, 0.14, 0.15]).validate()

    def test_the_wrong_number_of_horizons_is_rejected(self):
        with pytest.raises(ValueError, match="expected 6 horizons"):
            self.prediction([0.1, 0.2, 0.3]).validate()

    def test_as_dict_matches_the_results_schema(self):
        out = self.prediction([0.01, 0.02, 0.04, 0.06, 0.09, 0.12]).as_dict()
        assert [f"year_{h}" for h in HORIZONS] == [k for k in out if k.startswith("year_")]
        assert out["calibrated"] is False


class TestCalibrator:
    @staticmethod
    def synthetic(n: int = 3000, seed: int = 0):
        """Scores that rank well but are badly scaled — the case calibration fixes."""
        rng = np.random.default_rng(seed)
        labels = (rng.random(n) < 0.08).astype(int)
        true_p = np.where(labels == 1, rng.beta(6, 3, n), rng.beta(2, 8, n))
        # Squash into [0, 0.3]: ranking is preserved, the scale is wrong.
        scores = np.tile((true_p * 0.3).reshape(-1, 1), (1, len(HORIZONS)))
        return scores, labels

    def test_unfitted_is_an_explicit_pass_through(self):
        calibrator = Calibrator()
        assert not calibrator.fitted
        assert calibrator.apply([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_calibration_improves_the_brier_score(self):
        from sklearn.metrics import brier_score_loss

        scores, labels = self.synthetic()
        calibrator = Calibrator().fit(scores, labels)
        calibrated = np.array([calibrator.apply(list(row)) for row in scores])
        assert brier_score_loss(labels, calibrated[:, 0]) < brier_score_loss(labels, scores[:, 0])

    def test_calibration_never_reorders_patients(self):
        """Isotonic regression is monotone non-decreasing, so it cannot reorder.

        It can still *tie* patients together — pooling adjacent violators is
        exactly what it does — so ranking metrics may move up a little as
        anti-concordant pairs collapse into ties. What must never happen is a
        pair swapping order, which is what this asserts directly.
        """
        scores, labels = self.synthetic()
        calibrator = Calibrator().fit(scores, labels)
        calibrated = np.array([calibrator.apply(list(row)) for row in scores])

        order = np.argsort(scores[:, 0])
        mapped = calibrated[order, 0]
        assert np.all(np.diff(mapped) >= -1e-9), "calibration reordered two patients"

    def test_calibration_does_not_degrade_ranking(self):
        from sklearn.metrics import roc_auc_score

        scores, labels = self.synthetic()
        calibrator = Calibrator().fit(scores, labels)
        calibrated = np.array([calibrator.apply(list(row)) for row in scores])
        assert roc_auc_score(labels, calibrated[:, 0]) >= roc_auc_score(labels, scores[:, 0]) - 1e-9

    def test_platt_preserves_ranking_exactly(self):
        """Logistic calibration is strictly monotone, so AUC is unchanged."""
        from sklearn.metrics import roc_auc_score

        scores, labels = self.synthetic()
        calibrator = Calibrator(method="platt").fit(scores, labels)
        calibrated = np.array([calibrator.apply(list(row)) for row in scores])
        assert roc_auc_score(labels, calibrated[:, 0]) == pytest.approx(
            roc_auc_score(labels, scores[:, 0]), abs=1e-9
        )

    def test_output_stays_monotone_across_horizons(self):
        """Per-horizon fits can cross; the result must still be cumulative."""
        scores, labels = self.synthetic()
        calibrator = Calibrator().fit(scores, labels)
        curve = calibrator.apply([0.05, 0.04, 0.20, 0.03, 0.25, 0.10])
        assert curve == sorted(curve)
        assert all(0.0 <= v <= 1.0 for v in curve)

    def test_platt_also_works(self):
        scores, labels = self.synthetic()
        calibrator = Calibrator(method="platt").fit(scores, labels)
        assert calibrator.fitted
        assert all(0.0 <= v <= 1.0 for v in calibrator.apply(list(scores[0])))

    def test_fitting_needs_both_classes(self):
        scores, _ = self.synthetic(n=100)
        with pytest.raises(ValueError, match="both positive and negative"):
            Calibrator().fit(scores, np.zeros(100, dtype=int))

    def test_mismatched_lengths_are_rejected(self):
        scores, labels = self.synthetic(n=100)
        with pytest.raises(ValueError, match="labels"):
            Calibrator().fit(scores, labels[:50])

    def test_round_trips_through_disk(self, tmp_path):
        scores, labels = self.synthetic()
        original = Calibrator().fit(scores, labels)
        original.save(tmp_path / "cal.pkl")
        restored = Calibrator.load(tmp_path / "cal.pkl")
        assert restored.fitted
        assert restored.apply(list(scores[0])) == original.apply(list(scores[0]))

    def test_loading_a_missing_file_gives_an_unfitted_pass_through(self, tmp_path):
        assert not Calibrator.load(tmp_path / "absent.pkl").fitted


class TestCohortPercentile:
    def test_none_without_a_reference(self):
        assert cohort_percentile(0.04, None) is None
        assert cohort_percentile(0.04, np.array([])) is None

    def test_ranks_against_the_reference(self):
        reference = np.linspace(0.0, 1.0, 101)
        assert cohort_percentile(0.5, reference) == pytest.approx(50, abs=1)
        assert cohort_percentile(0.0, reference) == 0
        assert cohort_percentile(1.1, reference) == 100
