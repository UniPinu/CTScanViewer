"""Stage 4 — measure the system honestly on held-out data (§4 Stage 4, §6.3).

The protocol §6.3 calls non-negotiable:

  * training touches **only** `train`;
  * hyperparameters are chosen on `val`;
  * every headline number comes from `test`, computed once;
  * a guard fails the run if any test patient appears in the training manifest.

That guard runs first here, before a single metric, and raises rather than
warns. Leakage does not make results slightly optimistic — it makes them
meaningless, and a warning in a log is not proportionate to that.

**Accuracy is never reported.** At an ~8% positive rate (§1.3) a model that
predicts "no cancer" for everyone scores 92%, so accuracy would be a number that
rewards exactly the wrong behaviour. What is reported instead:

  ROC-AUC     ranking quality, comparable to Sybil's published ~0.92 at 1 year
  AUPRC       the honest metric under imbalance; its floor is the base rate,
              not 0.5, so it is always shown against that floor
  Brier       calibration and sharpness together, lower is better
  reliability predicted vs observed frequency, bucketed — the plot behind Brier
  C-index     Harrell's concordance over the 6-year horizon

Reference points from §1.4: Sybil reports ROC-AUC ≈ 0.92 at 1 year and a 6-year
C-index ≈ 0.75 on NLST. Numbers far above those on our own split should be read
as evidence of leakage rather than of success.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cohort.split import check_no_leakage
from core import jsonio, paths
from core.provenance import stamp
from ml.risk import HORIZONS


def concordance_index(scores: np.ndarray, events: np.ndarray) -> float:
    """Harrell's C-index: the fraction of comparable pairs ranked correctly.

    Implemented directly rather than pulled from `lifelines`, which would add a
    dependency for twenty lines. With a binary outcome and no censoring times,
    every positive/negative pair is comparable, and the statistic reduces to
    ROC-AUC computed pairwise — ties counting a half.
    """
    positives = scores[events == 1]
    negatives = scores[events == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    comparisons = positives[:, None] - negatives[None, :]
    concordant = float((comparisons > 0).sum())
    tied = float((comparisons == 0).sum())
    return (concordant + 0.5 * tied) / float(comparisons.size)


def reliability_curve(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> list[dict]:
    """Predicted probability vs observed frequency, in equal-width buckets.

    Empty buckets are omitted rather than reported as zero: at an 8% base rate
    most high-probability buckets contain nobody, and plotting them as observed
    frequency 0 would draw a calibration failure that never happened.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not inside.any():
            continue
        out.append({
            "bin": [round(float(lo), 3), round(float(hi), 3)],
            "n": int(inside.sum()),
            "predicted": round(float(scores[inside].mean()), 4),
            "observed": round(float(labels[inside].mean()), 4),
        })
    return out


def evaluate_horizon(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Every metric we report, for one time horizon."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    positives = int(labels.sum())
    base_rate = float(labels.mean()) if len(labels) else float("nan")
    result: dict = {
        "n": int(len(labels)),
        "n_positive": positives,
        "base_rate": round(base_rate, 4),
    }

    # AUC and AUPRC are undefined with a single class present.
    if positives == 0 or positives == len(labels):
        result["note"] = "only one class present; ranking metrics are undefined"
        return result

    result["roc_auc"] = round(float(roc_auc_score(labels, scores)), 4)
    result["auprc"] = round(float(average_precision_score(labels, scores)), 4)
    # AUPRC's floor is the base rate, so the lift over it is the real signal.
    result["auprc_lift"] = round(result["auprc"] / base_rate, 2) if base_rate else None
    result["brier"] = round(float(brier_score_loss(labels, scores)), 5)
    result["c_index"] = round(concordance_index(scores, labels), 4)
    result["reliability"] = reliability_curve(labels, scores)
    return result


def guard_against_leakage(predictions: pd.DataFrame, splits: pd.DataFrame,
                          evaluated_split: str = "test") -> None:
    """Fail loudly if the evaluated split overlaps the training manifest (§6.3)."""
    train_ids = splits.loc[splits["split"] == "train", "PatientID"].astype(str)
    held_out_ids = predictions["PatientID"].astype(str)
    check_no_leakage(train_ids, held_out_ids)

    declared = splits.set_index(splits["PatientID"].astype(str))["split"]
    mismatched = [
        pid for pid in held_out_ids
        if pid in declared.index and declared[pid] != evaluated_split
    ]
    if mismatched:
        raise AssertionError(
            f"{len(mismatched)} evaluated patient(s) are not in the '{evaluated_split}' split: "
            f"{mismatched[:10]}"
        )


def evaluate(predictions: pd.DataFrame, splits: pd.DataFrame, evaluated_split: str = "test") -> dict:
    """Score a table of predictions against labels. Runs the leakage guard first.

    `predictions` needs `PatientID` plus one column per horizon (`year_1` …
    `year_6`). Labels come from `splits`; patients whose label is unknown are
    dropped, because a prototype-track label set marks positives only and
    scoring the unknowns as negatives would fabricate the result (see
    `cohort/labels.py`).
    """
    guard_against_leakage(predictions, splits, evaluated_split)

    labelled = splits[splits["label"] >= 0].copy()
    labelled["PatientID"] = labelled["PatientID"].astype(str)
    merged = predictions.assign(PatientID=predictions["PatientID"].astype(str)).merge(
        labelled[["PatientID", "label", "label_source"]], on="PatientID", how="inner"
    )

    report: dict = {
        "split": evaluated_split,
        "n_predicted": int(len(predictions)),
        "n_labelled": int(len(merged)),
        "label_sources": merged["label_source"].value_counts().to_dict() if len(merged) else {},
        "horizons": {},
        "provenance": stamp("evaluation"),
        "leakage_guard": "passed",
    }

    if merged.empty:
        report["note"] = (
            "No evaluated patient carries a known label. With the prototype-track label "
            "sets this is expected: they list positives only, and the authoritative "
            "outcome column requires the CDAS application (CONTEXT.md §1.2). No metric "
            "is reported rather than a metric computed against assumed negatives."
        )
        return report

    labels = merged["label"].to_numpy(dtype=int)
    for horizon in HORIZONS:
        column = f"year_{horizon}"
        if column not in merged:
            continue
        report["horizons"][column] = evaluate_horizon(labels, merged[column].to_numpy(dtype=float))

    report["reference"] = {
        "sybil_published_roc_auc_year_1": 0.92,
        "sybil_published_c_index_6yr": 0.75,
        "note": "Published NLST figures from CONTEXT.md §1.4. Results far above these "
                "on our own split suggest leakage rather than improvement.",
    }
    return report


def format_report(report: dict) -> str:
    lines = [
        f"split            {report['split']}",
        f"predicted        {report['n_predicted']:,}",
        f"labelled         {report['n_labelled']:,}",
        f"leakage guard    {report['leakage_guard']}",
    ]
    if report.get("note"):
        lines += ["", report["note"]]
    if report["horizons"]:
        lines += ["", f"{'horizon':<10}{'n':>7}{'pos':>6}{'ROC-AUC':>10}{'AUPRC':>9}"
                      f"{'lift':>7}{'Brier':>9}{'C-idx':>8}"]
        for name, metrics in report["horizons"].items():
            if "roc_auc" not in metrics:
                lines.append(f"{name:<10}{metrics['n']:>7}{metrics['n_positive']:>6}"
                             f"   {metrics.get('note', '')}")
                continue
            lines.append(
                f"{name:<10}{metrics['n']:>7}{metrics['n_positive']:>6}"
                f"{metrics['roc_auc']:>10.4f}{metrics['auprc']:>9.4f}"
                f"{metrics['auprc_lift']:>7.2f}{metrics['brier']:>9.5f}{metrics['c_index']:>8.4f}"
            )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True,
                   help="parquet/CSV with PatientID and year_1..year_6 columns")
    p.add_argument("--splits", type=Path, default=paths.SPLITS_PARQUET,
                   help=f"splits table (default: {paths.SPLITS_PARQUET})")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="which split these predictions cover (default: test)")
    p.add_argument("--out", type=Path, default=None, help="write the report here (default: reports/eval_<split>.json)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.splits.exists():
        raise SystemExit(f"No splits at {args.splits} — run `python -m cohort.make_splits` first")

    read = pd.read_parquet if args.predictions.suffix in (".parquet", ".pq") else pd.read_csv
    predictions = read(args.predictions)
    splits = pd.read_parquet(args.splits)

    report = evaluate(predictions, splits, args.split)
    out = args.out or paths.REPORTS_DIR / f"eval_{args.split}.json"
    jsonio.write(out, report)

    print(format_report(report))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
