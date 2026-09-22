"""Deterministic, patient-level train/val/test assignment (CONTEXT.md §2.3).

The split is derived by hashing the patient ID. That is deliberately boring,
and it buys three things a `random.shuffle` cannot:

  * **No leakage.** A patient's T0/T1/T2 scans are near-duplicates of each
    other. Hashing the *patient* keeps every one of their scans on the same
    side of the wall. Shuffling per scan would put a baseline in train and its
    own follow-up in test, and quietly inflate every metric we report.
  * **Download-free.** Any patient's split is computable from their ID alone,
    before a single byte is fetched. A training run streams the cohort and
    skips whatever the hash says is val/test.
  * **Stable forever.** Same `seed` -> same test set, on any machine, in any
    order. Changing `seed` is the only way to get a different split, and it is
    a deliberate, visible act.

Two assignment modes:

  `assign_split`        pure hash. Streamable and ID-local, but the positive
                        rate in each split is only correct *in expectation*.
  `assign_stratified`   ranks patients by the same hash **within each label
                        stratum** and cuts at exact quantiles. Preserves the
                        ~8% positive rate exactly (§1.3), at the cost of
                        needing the whole label column in hand. Still fully
                        deterministic.

Both are reproducible; the pure hash is the source of truth for unlabelled
patients, and `make_splits.py` records which mode produced each row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

#: Bump this only to deliberately draw a new split. Never bump it to chase a metric.
DEFAULT_SEED = "nlst-v1"

#: train / val / test. Must sum to 1.
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)

SPLITS = ("train", "val", "test")

#: Hash resolution. 10_000 buckets is finer than any cohort we will ever split.
_BUCKETS = 10_000


def split_value(patient_id: str, seed: str = DEFAULT_SEED) -> float:
    """A stable pseudo-random number in [0, 1) for this patient.

    SHA-256 of "<seed>:<patient_id>", so it depends on nothing but its inputs —
    not on insertion order, not on the Python hash seed, not on the platform.
    """
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()
    return (int(digest, 16) % _BUCKETS) / _BUCKETS


def assign_split(
    patient_id: str,
    seed: str = DEFAULT_SEED,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> str:
    """Which split this patient belongs to. Pure function of the ID."""
    train_f, val_f, _ = _check_fractions(fractions)
    r = split_value(patient_id, seed)
    if r < train_f:
        return "train"
    return "val" if r < train_f + val_f else "test"


def assign_many(
    patient_ids: Iterable[str],
    seed: str = DEFAULT_SEED,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> pd.Series:
    """Vectorised `assign_split`, indexed by patient ID."""
    ids = [str(p) for p in patient_ids]
    return pd.Series([assign_split(p, seed, fractions) for p in ids], index=ids, name="split")


def assign_stratified(
    labels: pd.Series,
    seed: str = DEFAULT_SEED,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> pd.Series:
    """Exact-proportion split, computed independently inside each label stratum.

    `labels` is indexed by PatientID; its values are the strata (0 / 1 /
    "unknown" — anything hashable). Within a stratum the patients are ordered by
    the *same* hash as `assign_split`, then cut at exact quantiles, so each
    stratum contributes exactly the requested fractions. With ~2,050 positives
    against ~24,000 negatives (§1.3), that keeps the positive rate identical in
    train, val and test instead of merely close.

    Ties in the hash (possible, with 10k buckets and 26k patients) are broken by
    patient ID so the result stays order-independent.
    """
    train_f, val_f, _ = _check_fractions(fractions)
    out: dict[str, str] = {}

    for _, group in labels.groupby(labels.values, dropna=False):
        ids = sorted(str(p) for p in group.index)
        ordered = sorted(ids, key=lambda p: (split_value(p, seed), p))
        n = len(ordered)
        n_train = int(round(train_f * n))
        n_val = int(round(val_f * n))
        # Rounding can overshoot on tiny strata; test absorbs the slack.
        n_val = min(n_val, max(0, n - n_train))
        for i, pid in enumerate(ordered):
            out[pid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    return pd.Series(out, name="split").reindex([str(p) for p in labels.index])


def _check_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError(f"fractions must be (train, val, test), got {fractions!r}")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions!r} summing to {sum(fractions)}")
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative, got {fractions!r}")
    return tuple(float(f) for f in fractions)  # type: ignore[return-value]


def check_no_leakage(train_ids: Iterable[str], held_out_ids: Iterable[str]) -> None:
    """Raise if any patient appears on both sides of the wall (§6.3).

    Called by the evaluation entry point and by the test suite. It is cheap, and
    the failure mode it guards against is silent and catastrophic.
    """
    overlap = sorted(set(map(str, train_ids)) & set(map(str, held_out_ids)))
    if overlap:
        shown = ", ".join(overlap[:10])
        more = f" (+{len(overlap) - 10} more)" if len(overlap) > 10 else ""
        raise AssertionError(
            f"Split leakage: {len(overlap)} patient(s) are in both train and held-out: {shown}{more}"
        )
