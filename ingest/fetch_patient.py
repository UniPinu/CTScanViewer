"""Fetch every useful CT series for ONE patient and make it the viewer's data.

This is the *interactive* single-patient path — what the button in the app calls
(§4 Stage 1). The training pipeline uses `ingest/stream.py` directly; this script
wraps it with the extra behaviour a human demo needs: pick a patient, clear the
previous one, download, export for the browser, report progress as it goes.

Steps, in order:
  1. Read the cohort built by `cohort/build_cohort.py`, falling back to the
     frozen `manifest_*.s5cmd` if no cohort has been built yet.
  2. Pick a patient — random, a specific one, or a random *held-out test*
     patient so a demo can honestly say the model never trained on them.
  3. Delete the previous patient's DICOM and exported volumes.
  4. Download their series through the bounded cache.
  5. Export them for the viewer.

Only one patient's export lives under `viewer/public/data` at a time, so the
data directories stay out of git and anyone who clones the repo can press the
button and get a working dataset.

Usage:
    python -m ingest.fetch_patient                   # random patient
    python -m ingest.fetch_patient --patient 100012  # a specific one
    python -m ingest.fetch_patient --split test      # random held-out patient
    python -m ingest.fetch_patient --primary-only    # one series per round, not every kernel
    python -m ingest.fetch_patient --dry-run         # show the choice, download nothing
    python -m ingest.fetch_patient --json            # machine-readable progress (used by the viewer)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

from cohort.split import assign_split
from core import jsonio, paths
from ingest.stream import SeriesCache
from preprocess.export_volumes import export_series, write_index

JSON_MODE = False


def emit(stage: str, msg: str = "", **fields) -> None:
    """Report progress: one JSON object per line for the viewer, plain text otherwise."""
    if JSON_MODE:
        print(json.dumps({"stage": stage, "msg": msg, **fields}), flush=True)
    elif msg:
        print(msg, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series-table", type=Path, default=paths.SERIES_PARQUET,
                   help=f"cohort from build_cohort.py (default: {paths.SERIES_PARQUET})")
    p.add_argument("--patient", help="PatientID to fetch (default: random)")
    p.add_argument("--split", choices=["train", "val", "test"],
                   help="restrict the random choice to one split — use 'test' for honest demos")
    p.add_argument("--min-rounds", type=int, default=1,
                   help="only consider patients with at least this many screening rounds (default: 1)")
    p.add_argument("--primary-only", action="store_true",
                   help="fetch one series per round instead of every reconstruction")
    p.add_argument("--seed", type=int, default=None, help="seed for the random choice (default: time-based)")
    p.add_argument("--export-dir", type=Path, default=paths.VIEWER_DATA,
                   help=f"where viewer volumes go (default: {paths.VIEWER_DATA})")
    p.add_argument("--masks", action="store_true", help="also export a lung mask per series")
    p.add_argument("--keep-cache", action="store_true",
                   help="leave previously cached DICOM in place (default: clear it first)")
    p.add_argument("--dry-run", action="store_true", help="choose a patient and report, but download nothing")
    p.add_argument("--json", action="store_true", help="emit JSON progress lines instead of text")
    return p.parse_args(argv)


def load_cohort(series_table: Path) -> pd.DataFrame:
    """The cohort table, or a rebuilt one from the frozen manifest as a fallback."""
    if series_table.exists():
        return pd.read_parquet(series_table)

    emit("index", "No cohort yet — building one from the frozen manifest…")
    from cohort.build_cohort import annotate, read_manifest, select_primary

    manifest = paths.default_manifest()
    if manifest is None:
        raise SystemExit(
            f"No cohort at {series_table} and no manifest_*.s5cmd to fall back on.\n"
            "Run `python -m cohort.build_cohort` first."
        )
    df = select_primary(annotate(read_manifest(manifest, min_slices=10)))
    return df[df["is_axial"]]


def choose_patient(cohort: pd.DataFrame, args: argparse.Namespace) -> str:
    """Pick the patient to fetch, honouring --patient / --split / --min-rounds."""
    patients = cohort["PatientID"].astype(str)

    if args.patient:
        if args.patient not in set(patients):
            raise SystemExit(f"Patient {args.patient} has no usable series in this cohort")
        return args.patient

    eligible = pd.Index(sorted(patients.unique()))
    if args.min_rounds > 1:
        rounds = cohort[cohort["is_primary"]].groupby(patients)["round_key"].nunique()
        eligible = eligible.intersection(rounds[rounds >= args.min_rounds].index)
    if args.split:
        eligible = pd.Index([p for p in eligible if assign_split(p) == args.split])

    if len(eligible) == 0:
        raise SystemExit(
            f"No patient matches split={args.split} with >= {args.min_rounds} rounds"
        )
    return random.Random(args.seed).choice(list(eligible))


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    global JSON_MODE
    args = parse_args(argv)
    JSON_MODE = args.json

    emit("index", "Loading the cohort…")
    cohort = load_cohort(args.series_table)

    patient = choose_patient(cohort, args)
    chosen = cohort[cohort["PatientID"].astype(str) == patient]
    if args.primary_only:
        chosen = chosen[chosen["is_primary"]]
    chosen = chosen.sort_values(["round_key", "kernel"])

    total_mb = float(chosen["series_size_MB"].sum())
    n_rounds = int(chosen["round_key"].nunique())
    split = assign_split(patient)
    emit(
        "selected",
        f"Patient {patient} [{split}]: {len(chosen)} series over {n_rounds} round(s), {total_mb:.0f} MB",
        patient=patient, series=len(chosen), visits=n_rounds, rounds=n_rounds,
        mb=round(total_mb), split=split,
    )
    if not JSON_MODE:
        print(chosen[["round_key", "StudyDate", "kernel", "instanceCount", "series_size_MB"]]
              .to_string(index=False))
    if args.dry_run:
        return 0

    emit("clearing", f"Clearing {args.export_dir}")
    clear_dir(args.export_dir)

    cache = SeriesCache(on_progress=lambda stage, message: emit(stage, message))
    if not args.keep_cache:
        cache.evict_all()

    emit("download", f"Downloading {total_mb:.0f} MB…", done=0, total=round(total_mb * 1e6))

    entries: list[dict] = []
    fetched_mb = 0.0
    # Export each series as it arrives rather than afterwards: the cache is
    # allowed to evict an earlier series to make room for a later one, so the
    # DICOM is only guaranteed to be on disk right now (see SeriesCache.stream).
    for i, (series_dir, (_, row)) in enumerate(zip(cache.stream(chosen), chosen.iterrows()), 1):
        fetched_mb += float(row["series_size_MB"])
        emit("download", "", done=round(fetched_mb * 1e6), total=round(total_mb * 1e6))
        emit("export", f"Preparing {i}/{len(chosen)} for the viewer…", i=i, n=len(chosen))
        try:
            entries.append(export_series(series_dir, args.export_dir, with_mask=args.masks))
        except ValueError as e:
            # One unreadable series should not cost the whole patient.
            emit("log", f"skipped {row['SeriesInstanceUID'][-12:]}: {e}")

    if not entries:
        emit("error", f"Nothing could be exported for patient {patient}")
        return 1

    write_index(args.export_dir, entries)
    jsonio.write(args.export_dir / "patient.json", {
        "patient_id": patient,
        "split": split,
        "rounds": sorted({e["round"] for e in entries if e.get("round")}),
        "n_series": len(entries),
    })

    emit("done", f"Ready: patient {patient}, {len(entries)} scans", patient=patient,
         count=len(entries), split=split)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as e:
        if e.code not in (None, 0):
            emit("error", str(e.code))
        raise
    except Exception as e:  # noqa: BLE001 - report anything to the viewer, then fail
        emit("error", f"{type(e).__name__}: {e}")
        sys.exit(1)
