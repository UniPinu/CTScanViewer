"""Stage 0b — assign every patient to train / val / test (CONTEXT.md §2.3, §4).

Reads `series.parquet`, attaches whatever labels are available (see
`cohort/labels.py`), applies the deterministic patient-level hash split, and
writes `data/cohort/splits.parquet`:

    PatientID · split · label · label_source · n_rounds · n_series · split_mode

The hash in `cohort/split.py` stays the source of truth — this file is an
auditable, joinable *record* of it, not a second opinion. Re-running with the
same seed reproduces it byte for byte.

Usage:
    python -m cohort.make_splits                         # real labels, from IDC
    python -m cohort.make_splits --stratify              # exact positive rate per split
    python -m cohort.make_splits --labels extra.csv      # add another source
    python -m cohort.make_splits --seed nlst-v2          # deliberately draw a new split
    python -m cohort.make_splits --check                 # re-verify an existing splits.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cohort import labels as labels_mod
from cohort.split import (
    DEFAULT_FRACTIONS,
    DEFAULT_SEED,
    SPLITS,
    assign_many,
    assign_stratified,
    check_no_leakage,
)
from core import paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", type=Path, default=paths.SERIES_PARQUET,
                   help=f"cohort from build_cohort.py (default: {paths.SERIES_PARQUET})")
    p.add_argument("--out", type=Path, default=paths.SPLITS_PARQUET,
                   help=f"output parquet (default: {paths.SPLITS_PARQUET})")
    p.add_argument("--labels", type=Path, action="append", default=[],
                   help="extra label file (CSV/TSV/Parquet); repeatable, most-trusted source wins")
    p.add_argument("--no-idc-labels", action="store_true",
                   help="skip IDC's NLST participant table (see cohort/clinical.py)")
    p.add_argument("--seed", default=DEFAULT_SEED,
                   help=f"split seed — changing it draws a new split (default: {DEFAULT_SEED})")
    p.add_argument("--fractions", type=float, nargs=3, default=list(DEFAULT_FRACTIONS),
                   metavar=("TRAIN", "VAL", "TEST"), help="split fractions (default: 0.70 0.15 0.15)")
    p.add_argument("--stratify", action="store_true",
                   help="exact proportions within each label stratum (needs labels; see cohort/split.py)")
    p.add_argument("--check", action="store_true", help="verify an existing --out and exit")
    return p.parse_args(argv)


def build(series: pd.DataFrame, label_table: labels_mod.LabelTable, seed: str,
          fractions: tuple[float, float, float], stratify: bool) -> pd.DataFrame:
    """Join the cohort to its labels and assign each patient a split."""
    primary = series[series["is_primary"]] if "is_primary" in series else series
    per_patient = primary.groupby(primary["PatientID"].astype(str)).agg(
        # Rounds, not studies: a patient scanned twice in one round still had
        # one screening round (see build_cohort.round_key).
        n_rounds=("round_key", "nunique"),
        n_series=("SeriesInstanceUID", "nunique"),
    )

    frame = per_patient.join(label_table.frame, how="left")
    frame["label"] = frame["label"].fillna(labels_mod.UNKNOWN).astype(int)
    frame["label_source"] = frame["label_source"].fillna("none")

    if stratify:
        frame["split"] = assign_stratified(frame["label"], seed=seed, fractions=fractions)
        frame["split_mode"] = "stratified"
    else:
        frame["split"] = assign_many(frame.index, seed=seed, fractions=fractions)
        frame["split_mode"] = "hash"

    frame["split_seed"] = seed
    return frame.reset_index()[
        ["PatientID", "split", "label", "label_source", "n_rounds", "n_series", "split_mode", "split_seed"]
    ]


def verify(splits: pd.DataFrame) -> None:
    """The guard that §6.3 calls non-negotiable. Raises on any violation."""
    if splits["PatientID"].duplicated().any():
        dupes = splits.loc[splits["PatientID"].duplicated(), "PatientID"].tolist()[:5]
        raise AssertionError(f"A patient appears more than once in splits.parquet: {dupes}")

    train = splits.loc[splits["split"] == "train", "PatientID"]
    held_out = splits.loc[splits["split"].isin(["val", "test"]), "PatientID"]
    check_no_leakage(train, held_out)

    unexpected = set(splits["split"]) - set(SPLITS)
    if unexpected:
        raise AssertionError(f"Unknown split label(s): {sorted(unexpected)}")


def summarise(splits: pd.DataFrame) -> str:
    known = splits[splits["label"] >= 0]
    rows = []
    for name in SPLITS:
        part = splits[splits["split"] == name]
        part_known = known[known["split"] == name]
        positives = int((part_known["label"] == 1).sum())
        rate = f"{positives / len(part_known):6.2%}" if len(part_known) else "   n/a"
        rows.append(
            f"  {name:<6} {len(part):>7,} patients  {int(part['n_series'].sum()):>7,} series  "
            f"labelled {len(part_known):>6,}  positives {positives:>5,}  rate {rate}"
        )
    return "\n".join([
        f"patients          {len(splits):,}",
        f"mode              {splits['split_mode'].iloc[0]} (seed {splits['split_seed'].iloc[0]!r})",
        *rows,
    ])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.check:
        if not args.out.exists():
            raise SystemExit(f"No splits at {args.out} — run without --check first")
        splits = pd.read_parquet(args.out)
        verify(splits)
        print(f"{args.out} passes the leakage guard.\n")
        print(summarise(splits))
        return 0

    if not args.series.exists():
        raise SystemExit(f"No cohort at {args.series} — run `python -m cohort.build_cohort` first")

    series = pd.read_parquet(args.series)
    patient_ids = sorted(series["PatientID"].astype(str).unique())

    tables = [labels_mod.read_label_file(path) for path in args.labels]
    if not args.no_idc_labels:
        # The authoritative outcome, fetched alongside the imaging index. A few
        # MB and a couple of seconds, so it is the default rather than a flag.
        try:
            tables.append(labels_mod.from_idc(patient_ids))
        except Exception as e:  # noqa: BLE001 - degrade to unlabelled, do not fail
            print(f"warning: could not load IDC clinical labels ({type(e).__name__}: {e});\n"
                  f"         continuing with labels unknown.")

    label_table = labels_mod.merge(patient_ids, *tables)
    print(label_table.summary())

    if args.stratify and label_table.n_positive + label_table.n_negative == 0:
        raise SystemExit(
            "--stratify needs labels to stratify on, and every patient is unknown.\n"
            "Pass --labels, or drop --stratify to use the pure hash split."
        )

    splits = build(series, label_table, args.seed, tuple(args.fractions), args.stratify)
    verify(splits)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(summarise(splits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
