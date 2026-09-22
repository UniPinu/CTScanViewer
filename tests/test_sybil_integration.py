"""The risk head must reproduce Sybil's own published reference scores.

This is the test that proves the whole out-of-process seam is faithful: the
Python 3.10 subprocess, the JSON serialisation, the score ordering and the
`RiskPrediction` assembly. If any of it distorted the numbers, these would drift.

The reference values are the ones hard-coded in Sybil's own
`tests/regression_test.py::TestPredict::test_demo_data`, matched there against
their published demo series at `rel_tol=1e-6`. Ours currently match to 0.0 —
the identical float64 values — which is what you would expect when the only
thing between the model and the assertion is `json.dumps`.

Both the Sybil environment and the demo data are optional, so this skips rather
than fails when they are absent. To enable it:

    py -3.10 -m venv .venv-sybil
    .venv-sybil/Scripts/pip install -r requirements-sybil.txt
    # then fetch the demo series once, via Sybil's own regression test:
    #   SYBIL_TEST_RUN_REGRESSION=true python tests/regression_test.py
    # (from a clone of the Sybil repo; it lands in ~/.sybil/sybil_example)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml.risk import HORIZONS, RiskPrediction, load_backend, sybil_python

#: Sybil's published scores for its demo series, years 1 through 6.
#: Source: tests/regression_test.py in reginabarzilaygroup/Sybil @ d9fc81b.
REFERENCE_SCORES = [
    0.021628819563619374,
    0.03857256315036462,
    0.07191945816622261,
    0.07926975188037134,
    0.09584583525781108,
    0.13568094038444453,
]

#: Where Sybil's own test harness leaves the demo series.
DEMO_DATA = Path(os.path.expanduser("~/.sybil/sybil_example/sybil_demo_data"))

requires_sybil = pytest.mark.skipif(
    sybil_python() is None,
    reason="no Sybil environment; see requirements-sybil.txt",
)
requires_demo_data = pytest.mark.skipif(
    not DEMO_DATA.is_dir(),
    reason=f"Sybil demo series not found at {DEMO_DATA}",
)


@requires_sybil
def test_backend_reports_itself_available():
    backend = load_backend()
    report = backend.describe()
    assert report.get("ok") is True, report
    assert backend.available()
    # A CPU install still passes; this just records which one ran.
    assert report.get("torch")


@requires_sybil
@requires_demo_data
@pytest.mark.slow
def test_reproduces_sybil_reference_scores():
    """The end-to-end check: our wrapper must not perturb the model's output."""
    prediction = load_backend().predict(DEMO_DATA)

    assert isinstance(prediction, RiskPrediction)
    assert prediction.model == "sybil_ensemble"
    assert prediction.n_slices == 194, "the demo series is 194 slices"
    assert len(prediction.scores) == len(HORIZONS)

    for horizon, expected, actual in zip(HORIZONS, REFERENCE_SCORES, prediction.scores):
        assert actual == pytest.approx(expected, rel=1e-6), (
            f"year {horizon}: expected {expected!r}, got {actual!r}"
        )


@requires_sybil
@requires_demo_data
@pytest.mark.slow
def test_reference_scores_pass_our_own_validation():
    """The real model's output must satisfy the invariants we enforce.

    `RiskPrediction.validate` rejects probabilities outside [0, 1] and any
    decrease in cumulative risk. Those are properties of the quantity rather
    than of a particular model, so genuine Sybil output has to satisfy them —
    if it did not, the guard would be wrong, not the model.
    """
    load_backend().predict(DEMO_DATA).validate()


@requires_sybil
@requires_demo_data
@pytest.mark.slow
def test_attention_is_collated_onto_the_scan(tmp_path):
    """Attention must come back as a voxel-aligned volume, not raw logits.

    Sybil's raw tensors are (5, 1, 25, 256) logits over an internal grid. What
    crosses the seam has to be the collated (n_slices, 512, 512) volume, with
    non-negative values — the collation exponentiates, so anything negative
    means the raw logits leaked through instead.
    """
    import numpy as np

    prediction = load_backend().predict(DEMO_DATA, tmp_path)
    assert prediction.attention_dir is not None

    attention = np.load(tmp_path / "attention.npy").astype(np.float32)
    assert attention.ndim == 3
    assert attention.shape[0] == prediction.n_slices
    assert attention.shape[1:] == (512, 512)
    assert attention.min() >= 0.0, "attention should be exponentiated, not raw logits"
    assert attention.max() > 0.0, "attention is entirely zero"
