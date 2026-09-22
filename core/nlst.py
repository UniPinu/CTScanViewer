"""NLST dataset conventions, decoded once and shared by every stage.

NLST packs the acquisition parameters into `SeriesDescription` as a
comma-separated list, which is lucky: it means the cohort can be built, filtered
and ranked entirely from the IDC metadata index, without downloading a byte.

    "0,OPA,GE,LSPLUS,STANDARD,360,2.5,120,80,0.1,1.5"
     │  │   │  │      │        │   │   │   │
     │  │   │  │      │        │   │   │   └── tube current (mA)
     │  │   │  │      │        │   │   └────── tube voltage (kVp)
     │  │   │  │      │        │   └────────── slice thickness (mm)
     │  │   │  │      │        └────────────── reconstruction FOV (mm)
     │  │   │  │      └─────────────────────── convolution kernel
     │  │   │  └────────────────────────────── scanner model
     │  │   └───────────────────────────────── manufacturer code
     │  └───────────────────────────────────── OPA = axial series, OPL = localizer
     └──────────────────────────────────────── screening year: 0, 1 or 2 (T0/T1/T2)

Fields are literal "null"/"na" when the scanner did not record them, which is
why every accessor below returns None rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Screening round labels. NLST scanned at baseline plus two annual follow-ups.
ROUNDS = ("T0", "T1", "T2")

#: Reconstruction kernels grouped by how much they sharpen. Sharp kernels make
#: edges crisp but amplify noise; smooth ones are what density measurement and
#: Sybil-style models expect. Names are manufacturer-specific, hence the union.
SMOOTH_KERNELS = {
    "STANDARD", "SOFT", "A", "B",                    # GE / Philips
    "B10F", "B20F", "B30F", "B31F",                  # Siemens
    "FC01", "FC02", "FC03", "FC10",                  # Toshiba
}
SHARP_KERNELS = {
    "LUNG", "BONE", "DETAIL", "EDGE", "C", "D", "E", "YA",
    "B40F", "B45F", "B50F", "B60F", "B70F", "B80F",
    "FC30", "FC50", "FC51", "FC52", "FC53",
}

#: "OPA" series are the axial reconstructions we can actually view; "OPL" are
#: one- or two-frame scout/localizer images.
AXIAL_TAG = "OPA"

_MISSING = {"", "na", "null", "none", "nan"}


@dataclass(frozen=True)
class SeriesDescriptor:
    """Everything the NLST SeriesDescription tells us about one series."""

    raw: str
    study_year: int | None  # 0, 1, 2
    kind: str | None  # OPA / OPL
    manufacturer: str | None
    model: str | None
    kernel: str | None
    fov_mm: float | None
    slice_mm: float | None
    kvp: float | None
    ma: float | None

    @property
    def round_label(self) -> str | None:
        """'T0' / 'T1' / 'T2', the label used throughout the pipeline."""
        return ROUNDS[self.study_year] if self.study_year in (0, 1, 2) else None

    @property
    def is_axial(self) -> bool:
        """False for scouts/localizers, which are never worth downloading."""
        return self.kind == AXIAL_TAG

    @property
    def kernel_style(self) -> str:
        return kernel_style(self.kernel)


def parse_series_description(description: str | None) -> SeriesDescriptor:
    """Decode a NLST SeriesDescription. Unparseable fields come back as None."""
    raw = (description or "").strip()
    parts = [p.strip() for p in raw.split(",")]

    def text(i: int) -> str | None:
        if i >= len(parts) or parts[i].lower() in _MISSING:
            return None
        return parts[i]

    def number(i: int) -> float | None:
        value = text(i)
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    year = number(0)
    kind = text(1)
    return SeriesDescriptor(
        raw=raw,
        study_year=int(year) if year is not None and year in (0.0, 1.0, 2.0) else None,
        kind=kind.upper() if kind else None,
        manufacturer=text(2),
        model=text(3),
        kernel=text(4),
        fov_mm=number(5),
        slice_mm=number(6),
        kvp=number(7),
        ma=number(8),
    )


def kernel_style(kernel: str | None) -> str:
    """'smooth', 'sharp' or 'other' — the axis that matters for model input."""
    if not kernel:
        return "other"
    key = kernel.strip().upper()
    if key in SMOOTH_KERNELS:
        return "smooth"
    if key in SHARP_KERNELS:
        return "sharp"
    return "other"


def study_year_from_description(description: str | None) -> int | None:
    """Just the screening year — kept as a cheap helper for the viewer export."""
    return parse_series_description(description).study_year
