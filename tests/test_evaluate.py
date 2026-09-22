"""Evaluation must be correct under heavy class imbalance, and must refuse to
score anything that leaks (CONTEXT.md §1.3, §6.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluate import concordance_index, evaluate, evaluate_horizon, reliability_curve

HORIZON_COLUMNS = [f"year_{h}" for h in range(1, 7)]


def make_splits(n: int = 400, positive_rate: float = 0.08, split: str = "test") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = (rng.random(n) < positive_rate).astype(int)
    return pd.DataFrame({
        "PatientID": [str(1000 + i) for i in range(n)],
        "split": split,
        "label": labels,
        "label_source": "cdas",
        "n_rounds": 3,
    })


def make_predictions(splits: pd.DataFrame, signal: float) -> pd.DataFrame:
    """Scores correlated with the label by `signal` (0 = useless, 1 = perfect)."""
    rng = np.random.default_rng(1)
    noise = rng.random(len(splits))
    scores = np.clip(signal * splits["label"].to_numpy() + (1 - signal) * noise, 0, 1)
    frame = pd.DataFrame({"PatientID": splits["PatientID"]})
    for column in HORIZON_COLUMNS:
        frame[column] = scores
    return frame


class TestConcordance:
    def test_perfect_ranking_is_one(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        events = np.array([1, 1, 0, 0])
        assert concordance_index(scores, events) == pytest.approx(1.0)

    def test_inverted_ranking_is_zero(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        events = np.array([1, 1, 0, 0])
        assert concordance_index(scores, events) == pytest.approx(0.0)

    def test_all_ties_is_a_half(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        events = np.array([1, 1, 0, 0])
        assert concordance_index(scores, events) == pytest.approx(0.5)

    def test_matches_roc_auc(self):
        """With a binary outcome, Harrell's C-index and ROC-AUC coincide."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(3)
        events = (rng.random(300) < 0.1).astype(int)
        scores = np.clip(0.4 * events + 0.6 * rng.random(300), 0, 1)
        assert concordance_index(scores, events) == pytest.approx(roc_auc_score(events, scores), abs=1e-9)

    def test_single_class_is_nan(self):
        assert np.isnan(concordance_index(np.array([0.3, 0.6]), np.array([0, 0])))


class TestHorizonMetrics:
    def test_auprc_floor_is_the_base_rate(self):
        """A useless model's AUPRC should sit at the base rate, not at 0.5.

        This is the whole reason §1.3 insists on AUPRC over accuracy, so it is
        worth asserting rather than assuming.
        """
        rng = np.random.default_rng(5)
        labels = (rng.random(4000) < 0.08).astype(int)
        scores = rng.random(4000)
        metrics = evaluate_horizon(labels, scores)
        assert metrics["auprc"] == pytest.approx(labels.mean(), abs=0.03)
        assert metrics["auprc_lift"] == pytest.approx(1.0, abs=0.4)
        assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.05)

    def test_a_good_model_lifts_auprc_well_above_the_base_rate(self):
        rng = np.random.default_rng(6)
        labels = (rng.random(4000) < 0.08).astype(int)
        scores = np.clip(0.5 * labels + 0.5 * rng.random(4000), 0, 1)
        metrics = evaluate_horizon(labels, scores)
        assert metrics["roc_auc"] > 0.85
        assert metrics["auprc_lift"] > 3.0

    def test_accuracy_is_never_reported(self):
        labels = np.array([0] * 92 + [1] * 8)
        metrics = evaluate_horizon(labels, np.random.default_rng(7).random(100))
        assert "accuracy" not in metrics

    def test_single_class_is_reported_not_crashed(self):
        metrics = evaluate_horizon(np.zeros(50, dtype=int), np.random.default_rng(8).random(50))
        assert "roc_auc" not in metrics
        assert "one class" in metrics["note"]


class TestReliability:
    def test_empty_buckets_are_omitted(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.01, 0.02, 0.03, 0.04])  # all in the first bucket
        curve = reliability_curve(labels, scores, bins=10)
        assert len(curve) == 1
        assert curve[0]["n"] == 4

    def test_a_perfectly_calibrated_model_matches_observed(self):
        rng = np.random.default_rng(9)
        scores = rng.random(20000)
        labels = (rng.random(20000) < scores).astype(int)
        for bucket in reliability_curve(labels, scores, bins=10):
            assert bucket["observed"] == pytest.approx(bucket["predicted"], abs=0.03)

    def test_a_score_of_exactly_one_lands_in_the_last_bucket(self):
        curve = reliability_curve(np.array([1, 0]), np.array([1.0, 0.95]), bins=10)
        assert sum(b["n"] for b in curve) == 2


class TestLeakageGuard:
    def test_evaluation_refuses_when_a_test_patient_is_in_train(self):
        splits = make_splits(split="test")
        splits.loc[0, "split"] = "train"  # this patient is now in both
        predictions = make_predictions(splits, signal=0.5)
        with pytest.raises(AssertionError, match="leakage"):
            evaluate(predictions, splits, "test")

    def test_evaluation_refuses_a_patient_from_the_wrong_split(self):
        splits = make_splits(split="test")
        splits.loc[0, "split"] = "val"
        predictions = make_predictions(splits, signal=0.5)
        with pytest.raises(AssertionError, match="not in the 'test' split"):
            evaluate(predictions, splits, "test")

    def test_a_clean_split_passes_and_reports(self):
        splits = make_splits()
        report = evaluate(make_predictions(splits, signal=0.6), splits, "test")
        assert report["leakage_guard"] == "passed"
        assert report["horizons"]["year_1"]["roc_auc"] > 0.8


class TestUnknownLabels:
    def test_unknown_labels_are_dropped_not_treated_as_negative(self):
        """The prototype label sets list positives only (see cohort/labels.py).

        Scoring unknowns as negatives would invent ~25,000 healthy patients and
        produce a confident, meaningless AUPRC.
        """
        splits = make_splits()
        splits["label"] = -1
        report = evaluate(make_predictions(splits, signal=0.6), splits, "test")
        assert report["n_labelled"] == 0
        assert report["horizons"] == {}
        assert "CDAS" in report["note"]

    def test_partially_labelled_cohorts_score_only_the_labelled(self):
        splits = make_splits()
        splits.loc[splits.index[:300], "label"] = -1
        report = evaluate(make_predictions(splits, signal=0.6), splits, "test")
        assert report["n_labelled"] == 100
        assert report["horizons"]["year_1"]["n"] == 100
