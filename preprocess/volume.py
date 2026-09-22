"""Stage 2 — turn a raw DICOM series into a compact, model-ready volume (§4).

A downloaded series is 100–600 MB of per-slice DICOM. What every later stage
actually wants is one array of Hounsfield units plus the handful of facts needed
to interpret it. That conversion happens exactly once per series and is cached,
because it is pure: same DICOM in, same volume out.

Two things in here are load-bearing and easy to get silently wrong.

**Slice order.** DICOM files in a directory are in no particular order, and
`InstanceNumber` is not reliably anatomical. The only trustworthy ordering is
`ImagePositionPatient[2]`, the physical position along the patient's long axis.
Sorting ascending puts the abdomen first and the clavicles last, which is the
order Sybil expects (§2.1 calls this "the #1 silent-failure gotcha" — a flipped
volume produces a confident, meaningless prediction rather than an error).
`flipped_for_sybil` in the metadata records whether reordering was necessary.

**Slice spacing.** `SliceThickness` describes how thick each slice is, which is
not the same as how far apart their centres are — overlapping reconstructions
are common in NLST. Spacing is measured from consecutive positions and only
falls back to the tag when positions are missing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from core import jsonio
from core.nlst import parse_series_description

#: Below this, tissue is air. Used to find the lungs and to trim the HU range.
AIR_HU = -1000

#: Clip range kept for modelling: everything below is air, everything above is
#: bone or metal and carries no parenchymal information (§4 Stage 2).
HU_CLIP = (-1024, 400)

#: The display window the viewer opens on. Lung parenchyma lives around -600 HU.
LUNG_WINDOW = (-600, 1500)

#: Anything denser than this inside the chest is not lung air.
LUNG_THRESHOLD_HU = -320


@dataclass
class Series:
    """One CT series: the voxels, plus everything needed to interpret them."""

    hu: np.ndarray  # int16, [slice][row][col], ordered feet -> head
    spacing: tuple[float, float, float]  # mm per voxel, (z, y, x)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.hu.shape)  # type: ignore[return-value]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(float(n * s) for n, s in zip(self.hu.shape, self.spacing))  # type: ignore[return-value]


def read_slices(series_dir: Path) -> list[pydicom.Dataset]:
    """Read every DICOM in a directory, ordered feet -> head.

    Files that are not DICOM, or that carry no pixel data, are skipped rather
    than raised on: IDC downloads occasionally include a sidecar, and one stray
    file should not cost a whole series.
    """
    datasets = []
    for path in sorted(series_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(path)
        except (InvalidDicomError, OSError):
            continue
        if "PixelData" in ds:
            datasets.append(ds)

    if not datasets:
        raise ValueError(f"No readable DICOM slices in {series_dir}")
    return sort_slices(datasets)


def sort_slices(datasets: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    """Order slices along the patient's long axis, ascending (feet -> head)."""
    if all("ImagePositionPatient" in ds for ds in datasets):
        return sorted(datasets, key=lambda ds: float(ds.ImagePositionPatient[2]))
    warnings.warn(
        "Series is missing ImagePositionPatient; falling back to InstanceNumber, "
        "which is not guaranteed to be anatomical order",
        stacklevel=2,
    )
    return sorted(datasets, key=lambda ds: int(ds.get("InstanceNumber", 0)))


def slice_positions(datasets: list[pydicom.Dataset]) -> np.ndarray | None:
    if not all("ImagePositionPatient" in ds for ds in datasets):
        return None
    return np.array([float(ds.ImagePositionPatient[2]) for ds in datasets], dtype=np.float64)


def slice_spacing(datasets: list[pydicom.Dataset]) -> float:
    """Distance between neighbouring slice centres, in mm.

    Measured from positions where possible. `SliceThickness` is the fallback,
    not the default: NLST includes overlapping reconstructions where thickness
    and spacing genuinely differ, and using thickness there stretches the volume.
    """
    positions = slice_positions(datasets)
    if positions is not None and len(positions) > 1:
        gaps = np.diff(positions)
        spacing = float(np.median(np.abs(gaps)))
        if spacing > 0:
            return spacing
    return float(datasets[0].get("SliceThickness", 1.0) or 1.0)


def was_stored_head_first(datasets: list[pydicom.Dataset]) -> bool:
    """Whether the series' own instance order runs head -> feet.

    True means `read_slices` had to reverse it to satisfy Sybil's expected
    inferior-to-superior ordering. Recorded in metadata so a surprising
    prediction can be traced back to its input geometry.
    """
    numbered = [ds for ds in datasets if "InstanceNumber" in ds and "ImagePositionPatient" in ds]
    if len(numbered) < 2:
        return False
    by_instance = sorted(numbered, key=lambda ds: int(ds.InstanceNumber))
    z = [float(ds.ImagePositionPatient[2]) for ds in by_instance]
    # Compare first and last rather than every gap: a few non-monotonic slices
    # in the middle are a scanner quirk, not a reversed volume.
    return z[-1] < z[0]


def to_hounsfield(datasets: list[pydicom.Dataset]) -> np.ndarray:
    """Stack sorted slices into an int16 HU volume.

    Stored pixel values are scanner-specific integers; `RescaleSlope` and
    `RescaleIntercept` map them onto the Hounsfield scale, where water is 0 and
    air is -1000. Without this step, volumes from two scanners are not comparable
    — which would quietly break both the risk model and change detection.
    """
    first = datasets[0]
    rows, cols = int(first.Rows), int(first.Columns)
    out = np.empty((len(datasets), rows, cols), dtype=np.int16)

    for i, ds in enumerate(datasets):
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise ValueError(
                f"Slice {i} is {ds.Rows}x{ds.Columns}, but the series starts {rows}x{cols}; "
                "this directory holds more than one series"
            )
        slope = float(ds.get("RescaleSlope", 1) or 1)
        intercept = float(ds.get("RescaleIntercept", 0) or 0)
        pixels = ds.pixel_array.astype(np.float32)
        if slope != 1 or intercept != 0:
            pixels = pixels * slope + intercept
        # Some scanners pad outside the reconstruction circle with a value far
        # below air; clamping to int16 keeps the array small and the range sane.
        np.clip(pixels, -32768, 32767, out=pixels)
        out[i] = pixels.astype(np.int16)

    return out


def load_series(series_dir: Path, series_uid: str | None = None) -> Series:
    """Read a DICOM directory into a `Series` with full metadata."""
    datasets = read_slices(series_dir)
    first = datasets[0]
    hu = to_hounsfield(datasets)

    row_mm, col_mm = (float(v) for v in first.get("PixelSpacing", [1.0, 1.0]))
    spacing = (slice_spacing(datasets), row_mm, col_mm)
    described = parse_series_description(str(first.get("SeriesDescription", "")))

    meta = {
        "patient_id": str(first.get("PatientID", "")),
        "study_uid": str(first.get("StudyInstanceUID", "")),
        "series_uid": series_uid or str(first.get("SeriesInstanceUID", "")),
        "study_date": str(first.get("StudyDate", "")),
        "study_year": described.study_year,
        "round": described.round_label,
        "kernel": described.kernel,
        "kernel_style": described.kernel_style,
        "manufacturer": str(first.get("Manufacturer", "")),
        "model": str(first.get("ManufacturerModelName", "")),
        "description": described.raw,
        "shape": [int(n) for n in hu.shape],
        "spacing_mm": [round(float(s), 6) for s in spacing],
        "orientation": "IS",  # read_slices always delivers inferior -> superior
        "flipped_for_sybil": was_stored_head_first(datasets),
        "hu_range": [int(hu.min()), int(hu.max())],
        "n_slices": len(datasets),
    }
    return Series(hu=hu, spacing=spacing, meta=meta)


def clip_hu(hu: np.ndarray, limits: tuple[int, int] = HU_CLIP) -> np.ndarray:
    """Clamp to the range that carries tissue information (§4 Stage 2, step 3)."""
    return np.clip(hu, limits[0], limits[1])


def resample_isotropic(series: Series, target_mm: float = 1.0, order: int = 1) -> Series:
    """Resample onto a cubic voxel grid.

    Registration and any size measured in millimetres are only meaningful on a
    known grid, and NLST mixes 1–3 mm slice spacings across scanners. Linear
    interpolation is the right trade here: cubic would ring around the sharp
    air/tissue boundary that defines a nodule's edge.
    """
    from scipy import ndimage

    factors = tuple(float(s) / target_mm for s in series.spacing)
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return series

    resampled = ndimage.zoom(series.hu.astype(np.float32), factors, order=order)
    meta = dict(series.meta)
    meta["shape"] = [int(n) for n in resampled.shape]
    meta["spacing_mm"] = [target_mm, target_mm, target_mm]
    meta["resampled_from_mm"] = [round(float(s), 6) for s in series.spacing]
    return Series(hu=resampled.astype(np.int16), spacing=(target_mm, target_mm, target_mm), meta=meta)


def lung_mask(hu: np.ndarray, threshold: int = LUNG_THRESHOLD_HU) -> np.ndarray:
    """Segment the lungs, so change detection looks only at parenchyma (§4 step 4).

    A classical threshold-and-morphology segmentation rather than a learned one:
    it needs no weights, no GPU and no network, and at this threshold the lungs
    are the most obvious structure in a chest CT. The steps, and why each is
    needed:

      1. Everything below -320 HU is air — but that includes the room around the
         patient and the air gap in the scanner table.
      2. Discard air components touching the volume's side walls. Room air is
         one connected blob that reaches the edge; the lungs never do.
      3. Keep the two largest survivors. Usually the left and right lung;
         sometimes one component when the trachea joins them.
      4. Fill holes. Vessels, airway walls and any nodule are *denser* than the
         threshold, so they punch holes in the mask — and a nodule excluded from
         the lung mask is precisely the thing we must not drop.

    Returns a bool array shaped like `hu`.
    """
    from scipy import ndimage

    air = hu < threshold
    labels, count = ndimage.label(air)
    if count == 0:
        return np.zeros_like(hu, dtype=bool)

    # Room air and table air reach the left/right/front/back walls. The top and
    # bottom faces are excluded from this test: the lungs legitimately run off
    # the end of the scanned volume.
    border = np.unique(np.concatenate([
        labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ]))
    interior = np.ones(count + 1, dtype=bool)
    interior[0] = False
    interior[border[border > 0]] = False

    candidate_ids = np.flatnonzero(interior)
    if candidate_ids.size == 0:
        return np.zeros_like(hu, dtype=bool)

    sizes = ndimage.sum_labels(air, labels, index=candidate_ids)
    keep = candidate_ids[np.argsort(sizes)[::-1][:2]]
    # A second component far smaller than the first is a bowel-gas pocket or an
    # airway stub, not the other lung.
    largest = sizes.max()
    keep = [i for i in keep if ndimage.sum_labels(air, labels, index=i) > largest * 0.15]

    mask = np.isin(labels, keep)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))
    # Fill per slice: a 3-D fill would leak through the trachea in some volumes.
    for i in range(mask.shape[0]):
        mask[i] = ndimage.binary_fill_holes(mask[i])
    return mask


def save(series: Series, out_dir: Path, mask: np.ndarray | None = None) -> dict:
    """Write `volume.npy` (+ optional `mask.npy`) and `meta.json` for a series."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "volume.npy", series.hu)

    meta = dict(series.meta)
    meta["volume"] = "volume.npy"
    meta["hu_clip"] = list(HU_CLIP)
    if mask is not None:
        np.save(out_dir / "mask.npy", np.packbits(mask, axis=None))
        meta["lung_mask"] = "mask.npy"
        meta["lung_mask_shape"] = [int(n) for n in mask.shape]
        meta["lung_volume_ml"] = round(
            float(mask.sum()) * float(np.prod(series.spacing)) / 1000.0, 1
        )
    jsonio.write(out_dir / "meta.json", meta)
    return meta


def load_cached(out_dir: Path) -> tuple[Series, np.ndarray | None]:
    """Read back what `save` wrote."""
    meta = jsonio.read(out_dir / "meta.json")
    hu = np.load(out_dir / "volume.npy")
    series = Series(hu=hu, spacing=tuple(meta["spacing_mm"]), meta=meta)

    mask = None
    if meta.get("lung_mask"):
        shape = tuple(meta["lung_mask_shape"])
        packed = np.load(out_dir / "mask.npy")
        mask = np.unpackbits(packed, count=int(np.prod(shape))).astype(bool).reshape(shape)
    return series, mask
