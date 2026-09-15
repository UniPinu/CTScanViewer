"""Fetch every useful CT series for ONE patient and make it the viewer's data.

Steps, in order:
  1. Join the manifest with the IDC index and keep CT series with 10+ slices
     (drops the SEG/SR non-image series and the 1–2 slice localizers).
  2. Pick a patient — random, or the one given with --patient.
  3. Delete the previous patient's DICOM files and exported volumes.
  4. Download all of that patient's series (every visit, every kernel).
  5. Export them for the viewer (same as export_volumes.py).

Only one patient lives on disk at a time, so the data directories can stay
out of git and anyone who clones the repo can press the button in the app
(or run this script) to get a working dataset.

Usage:
    python fetch_patient.py                   # random patient
    python fetch_patient.py --patient 100012  # a specific one
    python fetch_patient.py --seed 42         # reproducible random choice
    python fetch_patient.py --dry-run         # show the choice, download nothing
    python fetch_patient.py --json            # machine-readable progress (used by the viewer)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from download_sample import DEFAULT_MANIFEST, UUID_RE, split_manifest
from export_volumes import export_series

DICOM_DIR = Path("data/patient")
EXPORT_DIR = Path("viewer/public/data")

JSON_MODE = False


def emit(stage: str, msg: str = "", **fields) -> None:
    """Report progress: one JSON object per line for the viewer, plain text otherwise."""
    if JSON_MODE:
        print(json.dumps({"stage": stage, "msg": msg, **fields}), flush=True)
    elif msg:
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="s5cmd manifest from IDC")
    p.add_argument("--patient", help="PatientID to fetch (default: random)")
    p.add_argument("--seed", type=int, default=None, help="seed for the random choice (default: time-based)")
    p.add_argument("--modality", default="CT", help="modality to keep, or 'all' (default: CT)")
    p.add_argument("--min-slices", type=int, default=10, help="skip series with fewer slices, i.e. localizers/scouts (default: 10)")
    p.add_argument("--dicom-dir", type=Path, default=DICOM_DIR, help=f"where DICOM files go (default: {DICOM_DIR})")
    p.add_argument("--export-dir", type=Path, default=EXPORT_DIR, help=f"where viewer volumes go (default: {EXPORT_DIR})")
    p.add_argument("--dry-run", action="store_true", help="choose a patient and report, but download nothing")
    p.add_argument("--json", action="store_true", help="emit JSON progress lines instead of text")
    return p.parse_args()


def usable_series(client, cp_lines: list[str], modality: str, min_slices: int) -> pd.DataFrame:
    """Manifest lines joined with the IDC index, filtered to the series worth downloading."""
    manifest = pd.DataFrame(
        {"cp_line": cp_lines, "crdc_series_uuid": [UUID_RE.search(l).group(1) for l in cp_lines]}
    )
    cols = ["crdc_series_uuid", "PatientID", "StudyDate", "Modality", "instanceCount", "series_size_MB"]
    df = manifest.merge(client.index[cols], on="crdc_series_uuid", how="inner")
    if modality.lower() != "all":
        df = df[df["Modality"] == modality.upper()]
    return df[df["instanceCount"] >= min_slices]


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    global JSON_MODE
    args = parse_args()
    JSON_MODE = args.json

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    emit("index", "Loading manifest and IDC index…")
    from idc_index import IDCClient  # slow import; do it after argument errors

    header, cp_lines = split_manifest(args.manifest)
    client = IDCClient()
    series = usable_series(client, cp_lines, args.modality, args.min_slices)
    patients = sorted(series["PatientID"].unique())
    if not patients:
        raise SystemExit("No usable series in the manifest")

    if args.patient:
        if args.patient not in patients:
            raise SystemExit(f"Patient {args.patient} has no usable series in this manifest")
        patient = args.patient
    else:
        patient = random.Random(args.seed).choice(patients)

    chosen = series[series["PatientID"] == patient].sort_values(["StudyDate", "series_size_MB"])
    total_mb = float(chosen["series_size_MB"].sum())
    n_visits = int(chosen["StudyDate"].nunique())
    emit(
        "selected",
        f"Patient {patient}: {len(chosen)} series over {n_visits} visit(s), {total_mb:.0f} MB "
        f"(from {len(patients)} candidates)",
        patient=patient, series=len(chosen), visits=n_visits, mb=round(total_mb), candidates=len(patients),
    )
    if not JSON_MODE:
        print(chosen[["StudyDate", "Modality", "instanceCount", "series_size_MB"]].to_string(index=False))
    if args.dry_run:
        return

    emit("clearing", f"Removing old data in {args.dicom_dir} and {args.export_dir}")
    clear_dir(args.dicom_dir)
    clear_dir(args.export_dir)

    manifest_path = args.dicom_dir / "manifest.s5cmd"
    manifest_path.write_text("\n".join(header + chosen["cp_line"].tolist()) + "\n")

    emit("download", f"Downloading {total_mb:.0f} MB…", done=0, total=round(total_mb * 1e6))
    last = [0.0]

    def on_progress(done_bytes: float, total_bytes: float, _unit: str, _desc: str) -> None:
        now = time.time()
        if now - last[0] >= 1.0:
            last[0] = now
            emit("download", "", done=int(done_bytes), total=int(total_bytes))

    client.download_from_manifest(
        manifestFile=str(manifest_path),
        downloadDir=str(args.dicom_dir),
        show_progress_bar=True,
        progress_callback=on_progress,
    )
    emit("download", "Download complete", done=round(total_mb * 1e6), total=round(total_mb * 1e6))

    series_dirs = sorted({f.parent for f in args.dicom_dir.rglob("*.dcm")})
    index = []
    for i, series_dir in enumerate(series_dirs, 1):
        n_files = sum(1 for _ in series_dir.glob("*.dcm"))
        if n_files < args.min_slices:
            continue
        emit("export", f"Exporting {i}/{len(series_dirs)}: {series_dir.name}", i=i, n=len(series_dirs))
        index.append(export_series(series_dir, args.export_dir))
    index.sort(key=lambda m: (m["patient"], m["studyDate"], m["kernel"]))
    (args.export_dir / "index.json").write_text(json.dumps(index, indent=2))

    emit("done", f"Ready: patient {patient}, {len(index)} scans", patient=patient, count=len(index))


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            emit("error", str(e))
        raise
    except Exception as e:  # noqa: BLE001 - report anything to the viewer, then fail
        emit("error", f"{type(e).__name__}: {e}")
        sys.exit(1)
