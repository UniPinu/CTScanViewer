"""The split is the one thing that must never quietly break (CONTEXT.md §6.3).

If a patient's baseline lands in train and their own follow-up in test, every
headline metric is inflated and nothing in the report means anything. These
tests are the guard, and they are meant to run in CI on every commit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cohort.split import (
    DEFAULT_FRACTIONS,
    DEFAULT_SEED,
    SPLITS,
    assign_many,
    assign_split,
    assign_stratified,
    check_no_leakage,
    split_value,
)

PATIENTS = [str(100000 + i) for i in range(4000)]


def test_split_value_is_in_range():
    assert all(0.0 <= split_value(p) < 1.0 for p in PATIENTS[:200])


def test_assignment_is_deterministic_across_calls():
    first = [assign_split(p) for p in PATIENTS]
    second = [assign_split(p) for p in PATIENTS]
    assert first == second


def test_assignment_does_not_depend_on_input_order():
    """The whole point of hashing: order-independence, on any machine."""
    forward = assign_many(PATIENTS)
    backward = assign_many(list(reversed(PATIENTS)))
    assert forward.to_dict() == backward.to_dict()


def test_assignment_is_stable_against_a_frozen_expectation():
    """Pin a few known IDs. If this fails, the test set silently changed.

    That is allowed to happen — but only by deliberately bumping DEFAULT_SEED,
    which should also update these values in the same commit.
    """
    frozen = {p: assign_split(p) for p in ["100002", "100004", "115345", "123342", "200511"]}
    assert frozen == {
        "100002": "train",  # hash 0.3404
        "100004": "train",  # hash 0.3098
        "115345": "train",  # hash 0.2727
        "123342": "val",    # hash 0.7072
        "200511": "train",  # hash 0.4313
    }


def test_every_patient_lands_in_exactly_one_split():
    assigned = assign_many(PATIENTS)
    assert len(assigned) == len(PATIENTS)
    assert set(assigned.unique()) <= set(SPLITS)


def test_fractions_are_roughly_respected():
    assigned = assign_many(PATIENTS)
    shares = assigned.value_counts(normalize=True)
    for name, expected in zip(SPLITS, DEFAULT_FRACTIONS):
        assert shares[name] == pytest.approx(expected, abs=0.02)


def test_changing_the_seed_changes_the_split():
    a = assign_many(PATIENTS, seed=DEFAULT_SEED)
    b = assign_many(PATIENTS, seed="nlst-v2")
    assert (a != b).sum() > len(PATIENTS) * 0.1


def test_all_of_a_patients_scans_share_one_split():
    """The leakage property itself, stated over a longitudinal cohort."""
    scans = pd.DataFrame(
        [{"PatientID": p, "round": r} for p in PATIENTS[:500] for r in ("T0", "T1", "T2")]
    )
    scans["split"] = [assign_split(p) for p in scans["PatientID"]]
    per_patient = scans.groupby("PatientID")["split"].nunique()
    assert (per_patient == 1).all()


def test_check_no_leakage_accepts_disjoint_sets():
    check_no_leakage(["1", "2", "3"], ["4", "5"])


def test_check_no_leakage_rejects_an_overlap():
    with pytest.raises(AssertionError, match="leakage"):
        check_no_leakage(["1", "2", "3"], ["3", "4"])


def test_check_no_leakage_ignores_id_type():
    """IDs arrive as str from parquet and int from some label files."""
    with pytest.raises(AssertionError):
        check_no_leakage([1, 2, 3], ["3"])


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        assign_split("100002", fractions=(0.5, 0.3, 0.3))


class TestStratified:
    """Exact-proportion mode, which is what the 8% positive rate needs (§1.3)."""

    @staticmethod
    def labels() -> pd.Series:
        # Deliberately lopsided, in NLST's proportions: ~8% positive.
        values = [1 if i % 12 == 0 else 0 for i in range(len(PATIENTS))]
        return pd.Series(values, index=PATIENTS, name="label")

    def test_positive_rate_is_preserved_in_every_split(self):
        labels = self.labels()
        assigned = assign_stratified(labels)
        overall = labels.mean()
        for name in SPLITS:
            part = labels[assigned == name]
            assert part.mean() == pytest.approx(overall, abs=0.005)

    def test_is_order_independent(self):
        labels = self.labels()
        forward = assign_stratified(labels)
        backward = assign_stratified(labels.iloc[::-1])
        assert forward.to_dict() == backward.to_dict()

    def test_keeps_each_patient_whole(self):
        assigned = assign_stratified(self.labels())
        assert assigned.index.is_unique
        assert set(assigned.unique()) <= set(SPLITS)

    def test_unknown_labels_form_their_own_stratum(self):
        labels = pd.Series([-1] * 100 + [1] * 50, index=PATIENTS[:150], name="label")
        assigned = assign_stratified(labels)
        for value in (-1, 1):
            part = assigned[labels == value]
            assert part.value_counts(normalize=True)["train"] == pytest.approx(0.70, abs=0.05)

    def test_tiny_strata_do_not_crash(self):
        labels = pd.Series([1, 0], index=["a", "b"], name="label")
        assigned = assign_stratified(labels)
        assert set(assigned.unique()) <= set(SPLITS)
