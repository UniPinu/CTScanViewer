"""Download a small sample of series from an IDC s5cmd manifest.

Each `cp` line in the manifest is one DICOM series (a full CT volume, i.e. a
folder of .dcm slices), so "5 images" here means 5 series. The manifest also
contains SEG (segmentation) and SR (structured report) series, which are
skipped by default so the sample is made up of actual CT scans.

Usage:
    python -m ingest.download_sample              # first 5 CT series -> data/sample
    python download_sample.py -n 10               # first 10 CT series
    python download_sample.py --random --seed 42  # 5 random CT series (reproducible)
    python download_sample.py --modality all      # don't filter by modality
    python download_sample.py --dry-run           # show what would be fetched, no download
    python download_sample.py --png               # download, then convert slices to PNG
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
from idc_index import IDCClient

from core import paths

# The manifest lives at the repo root, not beside this file — resolve it through
# core.paths so moving this module into ingest/ cannot break it again.
DEFAULT_MANIFEST = paths.default_manifest()
UUID_RE = re.compile(r"s3://[^/]+/([0-9a-f-]{36})/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="s5cmd manifest from IDC")
    p.add_argument("-n", "--count", type=int, default=5, help="number of series to download (default: 5)")
    p.add_argument("--out", type=Path, default=paths.SAMPLE_DIR, help=f"download directory (default: {paths.SAMPLE_DIR})")
    p.add_argument("--modality", default="CT", help="only keep series of this modality, or 'all' (default: CT)")
    p.add_argument("--random", action="store_true", help="pick a random sample instead of the first N lines")
    p.add_argument("--seed", type=int, default=0, help="RNG seed used with --random (default: 0)")
    p.add_argument("--dry-run", action="store_true", help="list the selected series and exit without downloading")
    p.add_argument("--png", action="store_true", help="after downloading, convert to PNG (see dicom_to_png.py)")
    return p.parse_args()


def split_manifest(path: Path) -> tuple[list[str], list[str]]:
    """Return (header comment lines, cp lines) from an s5cmd manifest."""
    header, cp_lines = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            header.append(line)
        elif "s3://" in line:
            cp_lines.append(line)
    return header, cp_lines


def lookup_series(client: IDCClient, cp_lines: list[str]) -> pd.DataFrame:
    """Join manifest cp lines with the IDC index, one row per line, in manifest order."""
    manifest = pd.DataFrame(
        {"cp_line": cp_lines, "crdc_series_uuid": [UUID_RE.search(l).group(1) for l in cp_lines]}
    )
    cols = ["crdc_series_uuid", "collection_id", "PatientID", "Modality", "series_size_MB"]
    return manifest.merge(client.index[cols], on="crdc_series_uuid", how="left")


def main() -> None:
    args = parse_args()
    if args.manifest is None or not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    header, cp_lines = split_manifest(args.manifest)
    print(f"Manifest has {len(cp_lines)} series. Loading IDC index ...")

    client = IDCClient()
    series = lookup_series(client, cp_lines)
    if args.modality.lower() != "all":
        series = series[series["Modality"] == args.modality.upper()]
        print(f"{len(series)} series with modality {args.modality.upper()}.")

    if series.empty:
        raise SystemExit("No series match; try --modality all")

    n = min(args.count, len(series))
    if args.random:
        idx = random.Random(args.seed).sample(list(series.index), n)
        selected = series.loc[sorted(idx)]
    else:
        selected = series.head(n)

    print(f"\nSelected {len(selected)} series:\n")
    print(selected.drop(columns="cp_line").to_string(index=False))
    print(f"\nTotal: {selected['series_size_MB'].sum():.1f} MB")

    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    sample_manifest = args.out / "sample_manifest.s5cmd"
    sample_manifest.write_text("\n".join(header + selected["cp_line"].tolist()) + "\n")
    print(f"\nWrote sample manifest to {sample_manifest}")
    print(f"Downloading to {args.out.resolve()} ...")

    client.download_from_manifest(
        manifestFile=str(sample_manifest),
        downloadDir=str(args.out),
        show_progress_bar=True,
    )
    print("Done.")

    if args.png:
        from preprocess.dicom_to_png import WINDOWS, convert_dir

        out_png = args.out.with_name(args.out.name + "_png")
        n = convert_dir(args.out, out_png, WINDOWS["lung"], bits=8, min_slices=2)
        print(f"Wrote {n} PNGs to {out_png.resolve()}")


if __name__ == "__main__":
    main()
