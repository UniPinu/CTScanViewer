"""Stage 3a — cancer risk from a single LDCT (CONTEXT.md §2.1, §6.1).

Sybil is a model trained on NLST that predicts 1–6-year lung-cancer risk from
one low-dose CT, with no clinical inputs and no manual annotation. §2.1 is
explicit that we stand on it rather than training a weaker model from scratch,
so this module is mostly plumbing around it, plus the calibration layer that
makes its probabilities honest on *our* cohort.

**Sybil runs in its own interpreter.** It pins Python <3.11, torch 1.13 and
numpy 1.24; this project runs on Python 3.13 with numpy 2.x. They cannot share a
process, so `SybilBackend` invokes `ml/sybil_runner.py` through the interpreter
named by `$SYBIL_PYTHON` and reads JSON back. Everything above that seam —
calibration, results assembly, the API, the UI — is unaffected by which
interpreter did the arithmetic.

**When Sybil is not installed, this module refuses to guess.** There is no
fallback that invents a number. §11 rules out anything that implies a prediction
the system did not make, and a plausible-looking fabricated risk is exactly the
failure a screening tool must never have. `predict` raises `RiskUnavailable`,
the API reports it as a first-class state, and the UI says so plainly. Change
detection, the cohort, the splits and the viewer all keep working meanwhile.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import paths

#: Horizons Sybil reports, in years.
HORIZONS = (1, 2, 3, 4, 5, 6)

#: Environment variable pointing at a Python 3.10 interpreter with Sybil installed.
SYBIL_PYTHON_ENV = "SYBIL_PYTHON"

#: Inference on a full series is slow the first time — the ensemble weights are
#: downloaded on first use.
DEFAULT_TIMEOUT_S = 1800


class RiskUnavailable(RuntimeError):
    """Raised when no risk model is installed.

    Deliberately a distinct type: callers must be able to tell "no model here"
    apart from "the model ran and failed", and must never paper over either with
    a default number.
    """


@dataclass
class RiskPrediction:
    """Risk at years 1–6, plus where it came from."""

    scores: list[float]  # cumulative risk at each horizon
    model: str
    n_slices: int = 0
    calibrated: bool = False
    attention_dir: Path | None = None
    attention_shapes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {f"year_{h}": round(float(s), 6) for h, s in zip(HORIZONS, self.scores)}
        out["curve"] = [round(float(s), 6) for s in self.scores]
        out["calibrated"] = self.calibrated
        return out

    def validate(self) -> None:
        """Risk must be a probability, and cumulative risk must not decrease.

        Both are properties of the quantity itself, not of any particular model.
        A violation means the output was misread — a transposed array, or the
        wrong tensor pulled out of the result — and is worth failing on rather
        than rendering.
        """
        if len(self.scores) != len(HORIZONS):
            raise ValueError(f"expected {len(HORIZONS)} horizons, got {len(self.scores)}")
        if not all(0.0 <= s <= 1.0 for s in self.scores):
            raise ValueError(f"risk scores outside [0, 1]: {self.scores}")
        drops = [
            (HORIZONS[i], self.scores[i], self.scores[i + 1])
            for i in range(len(self.scores) - 1)
            if self.scores[i + 1] < self.scores[i] - 1e-6
        ]
        if drops:
            raise ValueError(
                "cumulative risk decreases between horizons, which is impossible: "
                + ", ".join(f"year {y}: {a:.4f} -> {b:.4f}" for y, a, b in drops)
            )


# --------------------------------------------------------------------- backend


def sybil_python() -> str | None:
    """The interpreter that has Sybil, or None.

    Checks `$SYBIL_PYTHON`, then the conventional `.venv-sybil` beside the repo,
    so a standard setup needs no configuration at all.
    """
    configured = os.environ.get(SYBIL_PYTHON_ENV)
    if configured and Path(configured).exists():
        return configured

    for candidate in (
        paths.ROOT / ".venv-sybil" / "Scripts" / "python.exe",  # Windows
        paths.ROOT / ".venv-sybil" / "bin" / "python",  # POSIX
    ):
        if candidate.exists():
            return str(candidate)
    return None


@dataclass
class SybilBackend:
    """Runs Sybil out-of-process and parses its JSON."""

    model: str = "sybil_ensemble"
    python: str | None = field(default_factory=sybil_python)
    timeout_s: int = DEFAULT_TIMEOUT_S

    @property
    def name(self) -> str:
        return self.model

    def available(self) -> bool:
        return self.python is not None

    def describe(self) -> dict:
        """Environment report from the Sybil interpreter, for diagnostics."""
        if not self.python:
            return {"ok": False, "reason": self._missing_message()}
        try:
            out = self._invoke(["--selftest"], timeout_s=120)
            return json.loads(out)
        except Exception as e:
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    def predict(self, dicom_dir: Path, attention_dir: Path | None = None) -> RiskPrediction:
        if not self.python:
            raise RiskUnavailable(self._missing_message())

        argv = ["--dicom-dir", str(dicom_dir), "--model", self.model]
        if attention_dir:
            argv += ["--out", str(attention_dir)]
        else:
            argv.append("--no-attentions")

        payload = json.loads(self._invoke(argv, self.timeout_s))
        if "error" in payload:
            raise RuntimeError(f"Sybil failed: {payload['error']}")

        attention = payload.get("attention", {})
        prediction = RiskPrediction(
            scores=[float(v) for v in payload["scores_list"]],
            model=payload.get("model", self.model),
            n_slices=int(payload.get("n_slices", 0)),
            attention_dir=Path(attention["dir"]) if attention.get("dir") else None,
            attention_shapes={
                key: attention[f"{key}_shape"]
                for key in ("image_attention_1", "volume_attention_1")
                if f"{key}_shape" in attention
            },
        )
        prediction.validate()
        return prediction

    def _invoke(self, argv: list[str], timeout_s: int) -> str:
        """Run the runner script and return its stdout.

        The runner prints exactly one JSON object on stdout; anything it logs
        goes to stderr, which is surfaced only when the call fails.
        """
        command = [self.python, "-u", "-m", "ml.sybil_runner", *argv]
        proc = subprocess.run(
            command, cwd=paths.ROOT, capture_output=True, text=True, timeout=timeout_s,
        )
        stdout = proc.stdout.strip()
        if not stdout:
            raise RuntimeError(
                f"Sybil runner produced no output (exit {proc.returncode}).\n"
                f"stderr:\n{proc.stderr.strip()[-2000:]}"
            )
        # Torch and Sybil both chatter on stdout on first load; the JSON object
        # is the last line, so take that rather than the whole stream.
        return stdout.splitlines()[-1]

    def _missing_message(self) -> str:
        return (
            "No Sybil installation found.\n"
            f"Sybil needs its own Python 3.10 environment (it pins python<3.11, torch 1.13, "
            f"numpy 1.24), which cannot be the {sys.version.split()[0]} interpreter running "
            "this pipeline.\n"
            "Set one up:\n"
            "    py -3.10 -m venv .venv-sybil\n"
            "    .venv-sybil/Scripts/pip install sybil\n"
            f"then point ${SYBIL_PYTHON_ENV} at .venv-sybil/Scripts/python.exe, "
            "or leave it at that default path and it will be found automatically."
        )


@dataclass
class UnavailableBackend:
    """Stands in when nothing is installed. Always raises; never invents a score."""

    reason: str = "no risk model configured"

    @property
    def name(self) -> str:
        return "unavailable"

    def available(self) -> bool:
        return False

    def describe(self) -> dict:
        return {"ok": False, "reason": self.reason}

    def predict(self, dicom_dir: Path, attention_dir: Path | None = None) -> RiskPrediction:
        raise RiskUnavailable(self.reason)


def load_backend(model: str = "sybil_ensemble") -> SybilBackend | UnavailableBackend:
    """The risk backend for this machine."""
    backend = SybilBackend(model=model)
    if backend.available():
        return backend
    return UnavailableBackend(reason=backend._missing_message())


# ----------------------------------------------------------------- calibration


@dataclass
class Calibrator:
    """Maps raw model scores onto probabilities honest for our cohort (§6.1).

    Sybil is calibrated on its own training distribution. Once we select a
    cohort — one series per round, particular kernels, our own split — the
    scores can drift from observed frequencies. Isotonic regression, fit on the
    **train split only**, corrects that without assuming any functional form.

    Fitting one calibrator per horizon keeps the horizons independent, then
    `np.maximum.accumulate` restores the monotonicity that cumulative risk
    requires but per-horizon fitting does not guarantee.

    Fitting needs real labels, so this stays unfitted until the CDAS table lands
    (§1.2). An unfitted calibrator is an explicit pass-through, not a silent one.
    """

    models: dict[int, object] = field(default_factory=dict)
    method: str = "isotonic"
    n_train: int = 0

    @property
    def fitted(self) -> bool:
        return bool(self.models)

    def fit(self, scores: np.ndarray, labels: np.ndarray, horizons=HORIZONS) -> Calibrator:
        """Fit one calibrator per horizon. `scores` is (n_patients, n_horizons)."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if scores.ndim != 2 or scores.shape[1] != len(horizons):
            raise ValueError(f"scores must be (n, {len(horizons)}), got {scores.shape}")
        if len(labels) != len(scores):
            raise ValueError(f"{len(scores)} score rows but {len(labels)} labels")
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "calibration needs both positive and negative labels; got only "
                f"{np.unique(labels).tolist()}"
            )

        self.models = {}
        for i, horizon in enumerate(horizons):
            if self.method == "isotonic":
                model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                model.fit(scores[:, i], labels)
            else:  # Platt scaling
                model = LogisticRegression()
                model.fit(scores[:, i].reshape(-1, 1), labels)
            self.models[horizon] = model
        self.n_train = len(labels)
        return self

    def apply(self, scores: list[float], horizons=HORIZONS) -> list[float]:
        """Calibrate one patient's curve. Pass-through when unfitted."""
        if not self.fitted:
            return list(scores)

        out = []
        for horizon, score in zip(horizons, scores):
            model = self.models[horizon]
            if self.method == "isotonic":
                out.append(float(model.predict([score])[0]))
            else:
                out.append(float(model.predict_proba([[score]])[0, 1]))
        # Per-horizon fits can cross; cumulative risk cannot decrease.
        return [float(v) for v in np.maximum.accumulate(np.clip(out, 0.0, 1.0))]

    def save(self, path: Path) -> Path:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"models": self.models, "method": self.method, "n_train": self.n_train}, fh)
        return path

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        import pickle

        if not path.exists():
            return cls()
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        return cls(models=blob["models"], method=blob["method"], n_train=blob.get("n_train", 0))


def default_calibrator() -> Calibrator:
    return Calibrator.load(paths.ROOT / "models" / "calibrator.pkl")


def cohort_percentile(score: float, reference: np.ndarray | None) -> int | None:
    """Where this patient's 1-year risk sits in the screening cohort.

    Context matters more than the bare number: 2% means little on its own, but
    "higher than 84% of the screened cohort" is something a reader can act on.
    Returns None when no reference distribution has been computed, rather than
    implying a rank we have not earned.
    """
    if reference is None or len(reference) == 0:
        return None
    return int(round(float((np.asarray(reference) < score).mean() * 100)))


# ------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dicom-dir", type=Path, help="a directory holding ONE series")
    p.add_argument("--out", type=Path, default=None, help="where to write attention arrays")
    p.add_argument("--model", default="sybil_ensemble")
    p.add_argument("--check", action="store_true", help="report whether a risk model is available")
    args = p.parse_args(argv)

    backend = load_backend(args.model)

    if args.check or not args.dicom_dir:
        report = backend.describe()
        print(json.dumps(report, indent=2))
        if not report.get("ok"):
            # `reason` is set when no interpreter was found at all. When one was
            # found but could not load Sybil, the selftest reports the import
            # errors instead, and those are the actionable part.
            detail = report.get("reason") or "\n".join(
                f"{key}: {report[key]}" for key in ("sybil_error", "torch_error") if report.get(key)
            )
            print(f"\nrisk model: NOT AVAILABLE\n{detail}", file=sys.stderr)
            return 1
        device = "GPU" if report.get("cuda") else "CPU"
        print(f"\nrisk model: available (sybil {report.get('sybil')}, "
              f"torch {report.get('torch')}, {device})")
        return 0

    try:
        prediction = backend.predict(args.dicom_dir, args.out)
    except RiskUnavailable as e:
        print(f"risk model unavailable:\n{e}", file=sys.stderr)
        return 1

    for horizon, score in zip(HORIZONS, prediction.scores):
        print(f"  year {horizon}: {score:7.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
