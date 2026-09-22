"""Convert downloaded DICOM CT series into per-slice PNG files.

Walks the download directory, finds every folder containing .dcm files, sorts
the slices by position along the scan axis, converts pixel values to Hounsfield
units, applies a display window, and writes one PNG per slice. The output
mirrors the input folder hierarchy.

Slice reading, ordering and HU conversion all live in `preprocess/volume.py`;
this module is only the PNG rendering on top of them.

Usage:
    python -m preprocess.dicom_to_png                    # data/patient -> data/patient_png
    python -m preprocess.dicom_to_png --window soft      # soft-tissue window
    python -m preprocess.dicom_to_png --window -600 1500 # custom center/width
    python -m preprocess.dicom_to_png --bits 16          # lossless 16-bit PNG, no windowing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess.volume import load_series, read_slices, to_hounsfield  # noqa: F401

# Standard CT display windows as (center, width) in HU.
WINDOWS = {
    "lung": (-600, 1500),
    "soft": (40, 400),
    "bone": (400, 1800),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, default=Path("data/patient"), help="DICOM root (default: data/patient)")
    p.add_argument("--out", type=Path, default=None, help="PNG root (default: <in>_png)")
    p.add_argument(
        "--window", nargs="+", default=["lung"],
        help="preset name (lung, soft, bone) or 'CENTER WIDTH' in HU (default: lung)",
    )
    p.add_argument("--bits", type=int, choices=(8, 16), default=8, help="8 = windowed uint8, 16 = raw HU+1024 as uint16")
    p.add_argument("--min-slices", type=int, default=2, help="skip series with fewer slices, e.g. localizers (default: 2)")
    return p.parse_args()


def parse_window(spec: list[str]) -> tuple[float, float]:
    if len(spec) == 1 and spec[0] in WINDOWS:
        return WINDOWS[spec[0]]
    if len(spec) == 2:
        return float(spec[0]), float(spec[1])
    raise SystemExit(f"--window must be one of {list(WINDOWS)} or 'CENTER WIDTH', got {spec}")


def to_uint8(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width / 2, center + width / 2
    return (np.clip((hu - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def to_uint16(hu: np.ndarray) -> np.ndarray:
    # Shift so air (-1024 HU) maps to 0; keeps the full range losslessly.
    return np.clip(hu + 1024, 0, 65535).round().astype(np.uint16)


def convert_dir(src: Path, out: Path, window: tuple[float, float], bits: int, min_slices: int) -> int:
    """Convert every series under src; returns number of PNGs written."""
    series_dirs = sorted({f.parent for f in src.rglob("*.dcm")})
    written = 0
    for series_dir in series_dirs:
        n_files = sum(1 for _ in series_dir.glob("*.dcm"))
        if n_files < min_slices:
            print(f"skip {series_dir.name} ({n_files} slice)")
            continue
        hu = load_series(series_dir).hu
        img = to_uint8(hu, *window) if bits == 8 else to_uint16(hu)
        dest = out / series_dir.relative_to(src)
        dest.mkdir(parents=True, exist_ok=True)
        for i, sl in enumerate(img):
            Image.fromarray(sl).save(dest / f"{i:04d}.png")
        written += len(img)
        print(f"{series_dir.name}: {len(img)} slices -> {dest}")
    return written


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"Input directory not found: {args.src}")
    out = args.out or args.src.with_name(args.src.name + "_png")
    window = parse_window(args.window)
    n = convert_dir(args.src, out, window, args.bits, args.min_slices)
    print(f"\nWrote {n} PNGs to {out.resolve()}")


if __name__ == "__main__":
    main()
