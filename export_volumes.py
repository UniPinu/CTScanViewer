"""Export downloaded DICOM CT series as raw volumes for the web viewer.

For each series folder under the input directory this writes

    viewer/public/data/<id>/hu.i16     int16, little-endian, Hounsfield units,
                                       laid out [slice][row][col], feet -> head
    viewer/public/data/<id>/meta.json  shape, spacing, patient/visit/kernel info

and a viewer/public/data/index.json listing every exported series. The viewer
applies windowing itself, so the raw HU values are kept.

Usage:
    python export_volumes.py                 # data/sample -> viewer/public/data
    python export_volumes.py --in other_dir  # different DICOM root
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from dicom_to_png import read_slices, slices_to_hu

# Reconstruction kernels grouped by how much they sharpen (see context.md §6).
SHARP_KERNELS = {"B50f", "B60f", "B70f", "BONE", "LUNG"}
SMOOTH_KERNELS = {"B30f", "B20f", "STANDARD", "SOFT"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, default=Path("data/sample"), help="DICOM root (default: data/sample)")
    p.add_argument("--out", type=Path, default=Path("viewer/public/data"), help="output root (default: viewer/public/data)")
    p.add_argument("--min-slices", type=int, default=2, help="skip series with fewer slices (default: 2)")
    return p.parse_args()


def slice_spacing(slices) -> float:
    """Distance between neighbouring slices in mm."""
    if len(slices) > 1 and all("ImagePositionPatient" in s for s in slices):
        z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
        return float(np.median(np.diff(z)))
    return float(slices[0].get("SliceThickness", 1.0))


def visit_index(description: str) -> int | None:
    """NLST encodes the screening year as the first comma-separated field."""
    m = re.match(r"^(\d)\s*,", description or "")
    return int(m.group(1)) if m else None


def export_series(series_dir: Path, out_root: Path) -> dict:
    slices = read_slices(series_dir)
    hu = slices_to_hu(slices)
    first = slices[0]

    kernel = str(first.get("ConvolutionKernel", "") or "")
    style = "sharp" if kernel in SHARP_KERNELS else "smooth" if kernel in SMOOTH_KERNELS else "other"
    date = str(first.get("StudyDate", "") or "")
    uid = str(first.SeriesInstanceUID)
    series_id = f"{first.PatientID}_{date}_{kernel or 'NA'}_{uid[-6:]}"

    out_dir = out_root / series_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.clip(hu, -32768, 32767).astype("<i2").tofile(out_dir / "hu.i16")

    row_mm, col_mm = (float(v) for v in first.get("PixelSpacing", [1.0, 1.0]))
    meta = {
        "id": series_id,
        "patient": str(first.PatientID),
        "studyDate": f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date,
        "visit": visit_index(str(first.get("SeriesDescription", ""))),
        "kernel": kernel,
        "kernelStyle": style,
        "manufacturer": str(first.get("Manufacturer", "")),
        "model": str(first.get("ManufacturerModelName", "")),
        "description": str(first.get("SeriesDescription", "")),
        "shape": [int(n) for n in hu.shape],  # [slices, rows, cols]
        "spacing": [slice_spacing(slices), row_mm, col_mm],  # mm per voxel, same order
        "huRange": [int(hu.min()), int(hu.max())],
        "file": f"{series_id}/hu.i16",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"Input directory not found: {args.src}")

    series_dirs = sorted({f.parent for f in args.src.rglob("*.dcm")})
    index = []
    for series_dir in series_dirs:
        n_files = sum(1 for _ in series_dir.glob("*.dcm"))
        if n_files < args.min_slices:
            print(f"skip {series_dir.name} ({n_files} slice)")
            continue
        meta = export_series(series_dir, args.out)
        index.append(meta)
        d, h, w = meta["shape"]
        print(f"{meta['id']}: {d}x{h}x{w}  {d * h * w * 2 / 1e6:.0f} MB")

    index.sort(key=lambda m: (m["patient"], m["studyDate"], m["kernel"]))
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nExported {len(index)} series to {args.out.resolve()}")


if __name__ == "__main__":
    main()
