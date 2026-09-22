"""Fit the trainable parts of the pipeline — on the train split, and only that.

§6.1 sets the scope deliberately. Sybil was already trained on this
distribution, so retraining it from scratch buys nothing but compute. What is
worth training is the thin layer that makes its scores honest on *our* cohort:

  **MVP — calibration.** Isotonic (or Platt) regression mapping raw scores onto
  observed frequencies, fit on train, selected on val, reported on test. It is a
  small artifact, but it is a genuine trained-and-held-out-evaluated one, and it
  is what makes "4.1% risk" mean 4.1% rather than "a high score".

  **Stretch — fine-tuning Sybil's last block.** Documented in §6.1 and not done
  here: it needs the Sybil environment, real labels, and would be expected to
  produce only marginal gains.

Nothing in this file may read the val or test split for fitting. The guard at
the top enforces that rather than trusting it.

Usage:
    python -m ml.train --scores data/predictions/sybil_train.parquet
    python -m ml.train --scores ... --method platt --evaluate-on val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import jsonio, paths
from core.provenance import stamp
from ml.evaluate import evaluate_horizon
from ml.risk import HORIZONS, Calibrator

DEFAULT_MODEL_PATH = paths.ROOT / "models" / "calibrator.pkl"
SCORE_COLUMNS = [f"year_{h}" for h in HORIZONS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, required=True,
                   help="parquet/CSV of raw model scores: PatientID + year_1..year_6")
    p.add_argument("--splits", type=Path, default=paths.SPLITS_PARQUET,
                   help=f"splits table (default: {paths.SPLITS_PARQUET})")
    p.add_argument("--method", choices=["isotonic", "platt"], default="isotonic",
                   help="calibration family (default: isotonic)")
    p.add_argument("--evaluate-on", default="val", choices=["val", "test"],
                   help="split to report the fitted calibrator on (default: val)")
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH,
                   help=f"where to write the calibrator (default: {DEFAULT_MODEL_PATH})")
    return p.parse_args(argv)


def assert_train_only(frame: pd.DataFrame) -> None:
    """Refuse to fit on anything that is not the train split (§6.3)."""
    contaminants = sorted(set(frame["split"]) - {"train"})
    if contaminants:
        raise AssertionError(
            f"Calibration data contains non-train split(s): {contaminants}. "
            "Fitting on val or test would invalidate every number downstream."
        )


def join_labels(scores: pd.DataFrame, splits: pd.DataFrame, split: str) -> pd.DataFrame:
    """Scores joined to labels for one split, keeping only labelled patients."""
    labelled = splits[(splits["split"] == split) & (splits["label"] >= 0)].copy()
    labelled["PatientID"] = labelled["PatientID"].astype(str)
    joined = scores.assign(PatientID=scores["PatientID"].astype(str)).merge(
        labelled[["PatientID", "split", "label"]], on="PatientID", how="inner"
    )
    return joined


def fit(scores: pd.DataFrame, splits: pd.DataFrame, method: str) -> tuple[Calibrator, dict]:
    """Fit the calibrator on the train split."""
    training = join_labels(scores, splits, "train")
    if training.empty:
        raise SystemExit(
            "No labelled patients in the train split.\n"
            "Calibration needs real outcomes, which come from the CDAS Participant table "
            "(CONTEXT.md §1.2). Until it arrives, risk scores are reported uncalibrated "
            "and flagged as such in the UI."
        )
    assert_train_only(training)

    matrix = training[SCORE_COLUMNS].to_numpy(dtype=float)
    labels = training["label"].to_numpy(dtype=int)
    calibrator = Calibrator(method=method).fit(matrix, labels)

    return calibrator, {
        "n_train": int(len(training)),
        "n_train_positive": int(labels.sum()),
        "train_positive_rate": round(float(labels.mean()), 4),
        "method": method,
    }


def report_on(calibrator: Calibrator, scores: pd.DataFrame, splits: pd.DataFrame,
              split: str) -> dict:
    """Score the calibrator on a held-out split, before and after calibration."""
    held_out = join_labels(scores, splits, split)
    if held_out.empty:
        return {"split": split, "note": f"no labelled patients in '{split}'"}

    labels = held_out["label"].to_numpy(dtype=int)
    raw = held_out[SCORE_COLUMNS].to_numpy(dtype=float)
    calibrated = np.array([calibrator.apply(list(row)) for row in raw])

    out: dict = {"split": split, "n": int(len(held_out)), "horizons": {}}
    for i, column in enumerate(SCORE_COLUMNS):
        before = evaluate_horizon(labels, raw[:, i])
        after = evaluate_horizon(labels, calibrated[:, i])
        out["horizons"][column] = {
            # Calibration is monotone, so it cannot change the ranking — AUC and
            # AUPRC should be identical before and after. Brier is the number
            # that should improve, and showing both makes that visible.
            "roc_auc": before.get("roc_auc"),
            "brier_before": before.get("brier"),
            "brier_after": after.get("brier"),
            "reliability_after": after.get("reliability"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.splits.exists():
        raise SystemExit(f"No splits at {args.splits} — run `python -m cohort.make_splits` first")

    read = pd.read_parquet if args.scores.suffix in (".parquet", ".pq") else pd.read_csv
    scores = read(args.scores)
    missing = [c for c in ["PatientID", *SCORE_COLUMNS] if c not in scores.columns]
    if missing:
        raise SystemExit(f"{args.scores} is missing column(s): {missing}")

    splits = pd.read_parquet(args.splits)
    calibrator, info = fit(scores, splits, args.method)
    calibrator.save(args.out)

    report = {
        "calibration": info,
        "held_out": report_on(calibrator, scores, splits, args.evaluate_on),
        "provenance": stamp(f"calibrator-{args.method}"),
    }
    report_path = paths.REPORTS_DIR / f"calibration_{args.method}.json"
    jsonio.write(report_path, report)

    print(f"Fitted {args.method} calibration on {info['n_train']:,} train patients "
          f"({info['n_train_positive']:,} positive)")
    print(f"Wrote {args.out}")
    print(f"Wrote {report_path}")
    for column, metrics in report["held_out"].get("horizons", {}).items():
        print(f"  {column}: Brier {metrics['brier_before']} -> {metrics['brier_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
