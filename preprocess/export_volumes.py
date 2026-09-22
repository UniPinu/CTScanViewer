"""Export DICOM series as raw volumes the web viewer can read (§4 Stage 2).

For each series this writes

    viewer/public/data/<id>/hu.i16     int16, little-endian, Hounsfield units,
                                       laid out [slice][row][col], feet -> head
    viewer/public/data/<id>/meta.json  shape, spacing, provenance, split

plus a `viewer/public/data/index.json` listing every exported series. The
viewer applies windowing itself, so raw HU is kept rather than pre-rendered
greyscale — that is what lets the browser offer lung/soft-tissue/bone windows
without re-fetching anything.

The heavy lifting lives in `preprocess/volume.py`; this module is the viewer's
serialisation format and nothing more.

Usage:
    python -m preprocess.export_volumes                 # data/patient -> viewer/public/data
    python -m preprocess.export_volumes --in data/cache # a different DICOM root
    python -m preprocess.export_volumes --masks         # also export lung masks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cohort.split import assign_split
from core import jsonio, paths
from preprocess.volume import Series, load_series, lung_mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, default=paths.PATIENT_DIR,
                   help=f"DICOM root (default: {paths.PATIENT_DIR})")
    p.add_argument("--out", type=Path, default=paths.VIEWER_DATA,
                   help=f"output root (default: {paths.VIEWER_DATA})")
    p.add_argument("--min-slices", type=int, default=2, help="skip series with fewer slices (default: 2)")
    p.add_argument("--masks", action="store_true", help="also compute and export a lung mask per series")
    return p.parse_args(argv)


def series_id(series: Series) -> str:
    """A stable, human-readable directory name.

    Patient, date and kernel make it legible at a glance; the UID tail
    disambiguates the several reconstructions a study can contain.
    """
    meta = series.meta
    date = meta.get("study_date") or "NA"
    kernel = meta.get("kernel") or "NA"
    tail = (meta.get("series_uid") or "")[-6:] or "000000"
    return f"{meta.get('patient_id', 'NA')}_{date}_{kernel}_{tail}"


def viewer_meta(series: Series, sid: str, mask: np.ndarray | None = None) -> dict:
    """The JSON contract the React app consumes (see viewer/src/types.ts)."""
    meta = series.meta
    date = str(meta.get("study_date") or "")
    patient = str(meta.get("patient_id", ""))

    out = {
        "id": sid,
        "patient": patient,
        "studyDate": f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date,
        "visit": meta.get("study_year"),
        "round": meta.get("round"),
        "kernel": meta.get("kernel") or "",
        "kernelStyle": meta.get("kernel_style", "other"),
        "manufacturer": meta.get("manufacturer", ""),
        "model": meta.get("model", ""),
        "description": meta.get("description", ""),
        "shape": [int(n) for n in series.hu.shape],
        "spacing": [round(float(s), 6) for s in series.spacing],
        "huRange": [int(series.hu.min()), int(series.hu.max())],
        "file": f"{sid}/hu.i16",
        "seriesUid": meta.get("series_uid", ""),
        # Shown in the UI so a demo can honestly say "this is a held-out patient".
        "split": assign_split(patient) if patient else None,
        "flippedForSybil": bool(meta.get("flipped_for_sybil", False)),
    }
    if mask is not None:
        voxel_ml = float(np.prod(series.spacing)) / 1000.0
        out["lungVolumeMl"] = round(float(mask.sum()) * voxel_ml, 1)
        out["maskFile"] = f"{sid}/lung.u8"
    return out


def export_series(series_dir: Path, out_root: Path, with_mask: bool = False) -> dict:
    """Convert one DICOM series directory into the viewer's on-disk format."""
    series = load_series(series_dir)
    sid = series_id(series)
    out_dir = out_root / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    # int16 little-endian, which is what the browser reads straight into an
    # Int16Array with no per-voxel conversion.
    series.hu.astype("<i2").tofile(out_dir / "hu.i16")

    mask = None
    if with_mask:
        mask = lung_mask(series.hu)
        # One byte per voxel: the browser has no bit-unpacking primitive, and at
        # viewer resolution the size difference is not worth the JS to undo it.
        mask.astype(np.uint8).tofile(out_dir / "lung.u8")

    meta = viewer_meta(series, sid, mask)
    jsonio.write(out_dir / "meta.json", meta)
    return meta


def find_series_dirs(root: Path, min_slices: int) -> list[Path]:
    """Every directory under `root` that holds enough DICOM slices to be a series."""
    candidates = sorted({f.parent for f in root.rglob("*.dcm")})
    return [d for d in candidates if sum(1 for _ in d.glob("*.dcm")) >= min_slices]


def write_index(out_root: Path, entries: list[dict]) -> Path:
    """Write index.json, ordered so the viewer lists rounds chronologically."""
    entries.sort(key=lambda m: (m["patient"], m.get("studyDate") or "", m.get("kernel") or ""))
    return jsonio.write(out_root / "index.json", entries)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.src.exists():
        raise SystemExit(f"Input directory not found: {args.src}")

    series_dirs = find_series_dirs(args.src, args.min_slices)
    if not series_dirs:
        raise SystemExit(f"No DICOM series with >= {args.min_slices} slices under {args.src}")

    entries = []
    for series_dir in series_dirs:
        meta = export_series(series_dir, args.out, with_mask=args.masks)
        entries.append(meta)
        d, h, w = meta["shape"]
        extra = f"  lung {meta['lungVolumeMl']:.0f} mL" if "lungVolumeMl" in meta else ""
        print(f"{meta['id']}: {d}x{h}x{w}  {d * h * w * 2 / 1e6:.0f} MB{extra}")

    write_index(args.out, entries)
    print(f"\nExported {len(entries)} series to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
