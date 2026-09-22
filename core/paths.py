"""Every on-disk location the pipeline uses, in one place.

Stages are built independently (see CONTEXT.md §5) but they have to agree on
where things land. Importing from here is that agreement; nothing else should
hard-code a path.

Everything under `data/`, `results/` and `reports/` is git-ignored and
reproducible — deleting any of it costs time, never information.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Cohort: the queryable index of what exists, and who is in which split ---
COHORT_DIR = ROOT / "data" / "cohort"
SERIES_PARQUET = COHORT_DIR / "series.parquet"
SPLITS_PARQUET = COHORT_DIR / "splits.parquet"

# --- Ingestion: raw DICOM, bounded by an LRU cache ---
CACHE_DIR = ROOT / "data" / "cache"  # LRU-managed, one directory per series UID
PATIENT_DIR = ROOT / "data" / "patient"  # interactive single-patient path
SAMPLE_DIR = ROOT / "data" / "sample"  # download_sample.py's playground

# --- Preprocessing: compact model-ready volumes, computed once per series ---
VOLUME_DIR = ROOT / "data" / "volumes"  # <series_uid>/volume.npy + meta.json

# --- Outputs ---
VIEWER_DATA = ROOT / "viewer" / "public" / "data"  # what the browser fetches
RESULTS_DIR = ROOT / "results"  # results/<patient>.json, the viewer's payload
REPORTS_DIR = ROOT / "reports"  # held-out evaluation reports


def default_manifest() -> Path | None:
    """The frozen s5cmd manifest, kept as an offline fallback for the IDC query."""
    found = sorted(ROOT.glob("manifest_*.s5cmd"))
    return found[-1] if found else None


def cache_cap_bytes() -> int:
    """Hard ceiling on the DICOM cache. Override with CT_CACHE_GB."""
    return int(float(os.environ.get("CT_CACHE_GB", "50")) * 1e9)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
