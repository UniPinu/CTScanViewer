"""Stage 3c — turn model attention and change maps into things a viewer can draw.

Two overlays reach the UI: **attention**, meaning where the risk model looked,
and **change**, meaning where tissue densified between rounds. Both are produced
here in the same form, so the viewer has one overlay mechanism rather than two.

**Overlays ship as volumes, not as pre-rendered PNGs.** §5.3 sketches an
`overlay_dir` of images, and that would work, but the viewer already streams raw
HU volumes and windows them in the browser — which is what lets it offer an
opacity slider and three cutting planes without re-fetching anything. A stack of
baked PNGs would have to be re-rendered server-side on every opacity change and
would only exist for the axial plane. So `overlay_dir` still names a directory,
and the directory still holds what the schema promises plus a `meta.json`; what
is inside it is a `uint8` volume rather than an image per slice.

The rendering rule for attention, from §11: these maps show where the model
attended, not where disease is. The viewer labels them accordingly, and nothing
here should encourage reading them as a segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import jsonio

#: Overlays are quantised to a byte. The eye cannot resolve more through a
#: translucent colour ramp, and it keeps a whole-volume overlay a few MB.
OVERLAY_DTYPE = np.uint8

#: How many "most interesting" slices the results object advertises.
TOP_K_SLICES = 5


@dataclass
class Overlay:
    """A scalar field aligned to a series, ready for the browser to colour."""

    name: str  # "attention" | "change"
    data: np.ndarray  # uint8, [slice][row][col]
    shape: list[int]
    value_range: tuple[float, float]  # what 0 and 255 meant before quantising
    units: str
    top_slices: list[int] = field(default_factory=list)
    per_slice: list[float] = field(default_factory=list)
    #: Which screening round's geometry this overlay is in. An overlay drawn
    #: over a different round points at the wrong anatomy, so the viewer filters
    #: on this rather than relying on the shapes happening to differ.
    round: str | None = None
    #: The patient this overlay describes, checked by the viewer alongside the
    #: round. Belt and braces: `server/pipeline.infer` keeps the exported volumes
    #: and the overlays in step, and this catches it if they ever drift.
    patient_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "file": f"{self.name}.u8",
            "shape": self.shape,
            "value_range": [round(float(v), 2) for v in self.value_range],
            "units": self.units,
            "top_slices": self.top_slices,
            "per_slice": [round(float(v), 4) for v in self.per_slice],
            "round": self.round,
            "patient_id": self.patient_id,
        }

    def save(self, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.data.tofile(out_dir / f"{self.name}.u8")
        meta = self.as_dict()
        jsonio.write(out_dir / "meta.json", meta)
        return meta


def quantise(field_values: np.ndarray, lo: float | None = None,
             hi: float | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale a float field to uint8, returning the range that was mapped.

    The range is carried alongside so the viewer's legend can show real units
    rather than "0–255", and so two patients' overlays are never compared on a
    scale that silently differs between them.
    """
    values = np.nan_to_num(field_values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(values.min()) if lo is None else float(lo)
    hi = float(values.max()) if hi is None else float(hi)
    if hi <= lo:
        return np.zeros(values.shape, dtype=OVERLAY_DTYPE), (lo, lo)
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return scaled.round().astype(OVERLAY_DTYPE), (lo, hi)


def top_slices(per_slice: np.ndarray, k: int = TOP_K_SLICES, min_separation: int = 5) -> list[int]:
    """The k most-activated slices, spread out.

    Taking the raw top-k returns five adjacent slices through one structure,
    which tells a reader nothing they could not see by scrolling. Enforcing a
    minimum separation makes the list a set of distinct places to look.
    """
    order = np.argsort(per_slice)[::-1]
    chosen: list[int] = []
    for index in order:
        if per_slice[index] <= 0:
            break
        if all(abs(int(index) - c) >= min_separation for c in chosen):
            chosen.append(int(index))
        if len(chosen) == k:
            break
    return chosen


# ------------------------------------------------------------------- attention


def attention_overlay(attention_dir: Path, shape: tuple[int, int, int]) -> Overlay | None:
    """Build an overlay from the attention volume `ml/sybil_runner.py` saved.

    The runner writes `attention.npy` already collated onto the scan's own
    geometry — `(n_slices, 512, 512)`, one attention value per voxel. That
    deliberately keeps Sybil's architecture on Sybil's side of the seam: the raw
    tensors are per-ensemble-member logits over a 25x16x16 internal grid, and
    interpreting them here would mean re-deriving the model's own collation and
    getting it subtly wrong.

    So all that is left is geometry: crop or resample onto the volume the viewer
    is actually showing, and quantise.
    """
    from scipy import ndimage

    path = attention_dir / "attention.npy"
    if not path.exists():
        return None

    field_values = np.load(path).astype(np.float32)
    if field_values.ndim != 3:
        raise ValueError(
            f"{path.name} should be a 3-D attention volume, got shape {field_values.shape}"
        )

    # Sybil renders at the DICOM's native 512x512 with one plane per slice, which
    # is usually exactly the viewer's grid; resample only when it is not.
    if tuple(field_values.shape) != tuple(shape):
        factors = tuple(target / source for target, source in zip(shape, field_values.shape))
        field_values = ndimage.zoom(field_values, factors, order=1)
        field_values = field_values[: shape[0], : shape[1], : shape[2]]

    data, value_range = quantise(field_values)
    # Sum, not max: attention is already normalised per slice by the volume
    # weight, so total attention is what distinguishes an interesting slice.
    per_slice = field_values.reshape(field_values.shape[0], -1).sum(axis=1)

    return Overlay(
        name="attention",
        data=data,
        shape=[int(n) for n in data.shape],
        value_range=value_range,
        units="model attention (relative)",
        top_slices=top_slices(per_slice),
        per_slice=(per_slice / max(float(per_slice.max()), 1e-9)).tolist(),
    )


# ---------------------------------------------------------------------- change


def change_overlay(delta: np.ndarray, mask: np.ndarray | None = None,
                   hu_range: tuple[float, float] = (100.0, 800.0)) -> Overlay:
    """Build an overlay from a registered difference volume.

    Only densification is shown, on a fixed 100–800 HU scale rather than a
    per-patient one. A fixed scale means the same colour means the same
    magnitude on every patient — auto-scaling would make a noisy scan with no
    findings look exactly as alarming as one with a real nodule.
    """
    positive = np.clip(delta.astype(np.float32), 0.0, None)
    if mask is not None:
        positive = positive * mask
    data, value_range = quantise(positive, lo=hu_range[0], hi=hu_range[1])
    per_slice = positive.reshape(positive.shape[0], -1).sum(axis=1)

    return Overlay(
        name="change",
        data=data,
        shape=[int(n) for n in data.shape],
        value_range=value_range,
        units="HU increase",
        top_slices=top_slices(per_slice),
        per_slice=(per_slice / max(per_slice.max(), 1e-6)).tolist(),
    )


def resample_overlay_to(overlay: Overlay, shape: tuple[int, int, int]) -> Overlay:
    """Put an overlay onto a different grid, e.g. the viewer's native resolution.

    Change detection runs on a 1.5 mm isotropic grid while the viewer shows the
    original series geometry. Nearest-neighbour keeps the overlay's edges where
    they are rather than inventing intermediate values.
    """
    from scipy import ndimage

    if tuple(overlay.shape) == tuple(shape):
        return overlay
    factors = tuple(target / source for target, source in zip(shape, overlay.shape))
    resized = ndimage.zoom(overlay.data, factors, order=0).astype(OVERLAY_DTYPE)
    resized = resized[: shape[0], : shape[1], : shape[2]]

    # Resample the per-slice profile rather than recomputing it from the bytes.
    # Recomputing would have to reduce a saturated uint8 volume to one number per
    # slice, and every reduction is wrong here: `max` ties at 255 across most
    # slices and collapses the ranking into index order, while `sum` over
    # quantised data is not the same quantity the original profile measured.
    # Interpolating the profile keeps whatever ranking the overlay was built with.
    source_profile = np.asarray(overlay.per_slice, dtype=np.float32)
    if source_profile.size:
        per_slice = ndimage.zoom(source_profile, shape[0] / source_profile.size, order=1)
        per_slice = np.resize(per_slice, shape[0])
    else:
        per_slice = np.zeros(shape[0], dtype=np.float32)

    return Overlay(
        name=overlay.name,
        data=resized,
        shape=[int(n) for n in resized.shape],
        value_range=overlay.value_range,
        units=overlay.units,
        top_slices=top_slices(per_slice),
        per_slice=(per_slice / max(float(per_slice.max()), 1e-6)).tolist(),
        round=overlay.round,
        patient_id=overlay.patient_id,
    )
