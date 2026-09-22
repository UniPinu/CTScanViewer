"""Where the cancer label for each patient comes from (CONTEXT.md §1.2).

§1.2 assumes the authoritative outcome is locked behind a CDAS application and
plans a prototype track around waiting for it. That turned out to be unnecessary:
IDC publishes the NLST participant table next to the imaging index, so the real
label — positives *and* negatives, with diagnosis timing, stage and histology —
is available immediately. `cohort/clinical.py` reads it.

This module stays the seam between label sources, because more than one exists
and they are not equally trustworthy:

  **IDC clinical tables** (the normal path, and available today) — the NLST
  participant table, served alongside the imaging index. Positives *and*
  negatives for every participant, so no guessing is involved. See
  `cohort/clinical.py`; this is what §1.2 expected to need a CDAS application.

  **Annotation sets** (Sybil lesion boxes, NLSTseg masks) — *positives only*.
  Useful for localisation work, but as a label source they say nothing about
  anyone they do not list. A patient absent from them is unannotated, not
  healthy, and treating that as a negative would manufacture ~25,000 fake
  negatives and make every metric meaningless.

  **A CDAS extract**, if one is obtained, still takes precedence.

`label` is therefore three-valued throughout the pipeline:

    1        confirmed cancer
    0        no diagnosis on record, with a follow-up duration to back it up
   -1        unknown — no clinical record at all

Downstream, `ml/train.py` and `ml/evaluate.py` drop `-1` rows rather than
guessing. That means the prototype track can build, wire and demo everything
end-to-end while reporting *no* supervised metric it has not earned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UNKNOWN = -1
NEGATIVE = 0
POSITIVE = 1

#: Ordered by trust. When two sources disagree, the earlier one wins.
#: `idc` sits alongside `cdas` because it *is* the CDAS participant table, served
#: through IDC without the application (see cohort/clinical.py).
SOURCE_PRIORITY = ("cdas", "idc", "nlstseg", "sybil", "manual", "none")


@dataclass(frozen=True)
class LabelTable:
    """Per-patient labels plus a note on where each one came from."""

    frame: pd.DataFrame  # index: PatientID; columns: label, label_source

    @property
    def n_positive(self) -> int:
        return int((self.frame["label"] == POSITIVE).sum())

    @property
    def n_negative(self) -> int:
        return int((self.frame["label"] == NEGATIVE).sum())

    @property
    def n_unknown(self) -> int:
        return int((self.frame["label"] == UNKNOWN).sum())

    def summary(self) -> str:
        total = len(self.frame)
        known = self.n_positive + self.n_negative
        rate = f"{self.n_positive / known:.1%}" if known else "n/a"
        by_source = self.frame["label_source"].value_counts().to_dict()
        return (
            f"{total:,} patients: {self.n_positive:,} positive, {self.n_negative:,} negative, "
            f"{self.n_unknown:,} unknown (positive rate among known: {rate})\n"
            f"sources: {by_source}"
        )


def from_idc(patient_ids: list[str] | None = None) -> pd.DataFrame:
    """Labels from IDC's NLST participant table — the authoritative source.

    Returned in the same shape as `read_label_file`, so it composes with any
    other source through `merge`.
    """
    from cohort import clinical

    frame = clinical.outcomes().set_index("PatientID")[["label", "label_source"]]
    if patient_ids is not None:
        frame = frame.reindex([str(p) for p in patient_ids]).dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)
    frame.index.name = "PatientID"
    return frame


def empty(patient_ids: list[str]) -> LabelTable:
    """Everyone unknown — the prototype-track default before any file is supplied."""
    frame = pd.DataFrame(
        {"label": UNKNOWN, "label_source": "none"},
        index=pd.Index([str(p) for p in patient_ids], name="PatientID"),
    )
    return LabelTable(frame)


def read_label_file(path: Path, source: str | None = None) -> pd.DataFrame:
    """Read a label file into the canonical (PatientID, label, label_source) shape.

    Accepts CSV, TSV or Parquet with a patient column named any of `PatientID`,
    `patient_id`, `pid` or `patient`, and a label column named `label`,
    `cancer`, `canc_yr` or `confirmed`. That covers the shapes the CDAS extract
    and the published annotation sets actually arrive in, without asking anyone
    to hand-rename columns first.

    A file with no label column at all is read as a positives-only list — which
    is exactly what the Sybil and NLSTseg annotation sets are.
    """
    if path.suffix.lower() in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
    else:
        sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
        raw = pd.read_csv(path, sep=sep, dtype=str)

    patient_col = _first_column(raw, ["PatientID", "patient_id", "pid", "patient"])
    if patient_col is None:
        raise ValueError(
            f"{path}: no patient column found (looked for PatientID / patient_id / pid / patient); "
            f"columns present: {list(raw.columns)}"
        )

    label_col = _first_column(raw, ["label", "cancer", "canc_yr", "confirmed", "conflc"])
    out = pd.DataFrame(index=pd.Index(raw[patient_col].astype(str).str.strip(), name="PatientID"))
    if label_col is None:
        # Positives-only annotation list: presence in the file *is* the label.
        out["label"] = POSITIVE
    else:
        out["label"] = raw[label_col].map(_coerce_label).to_numpy()
    out["label_source"] = source or _infer_source(path)
    return out[~out.index.duplicated(keep="first")]


def merge(patient_ids: list[str], *tables: pd.DataFrame) -> LabelTable:
    """Combine label tables, most-trusted source winning, over the full cohort.

    Patients in the cohort but absent from every table come back `unknown`;
    patients present in a table but absent from the cohort are dropped, so the
    result always lines up with `series.parquet`.
    """
    result = empty(patient_ids).frame
    for table in sorted(tables, key=lambda t: _source_rank(t["label_source"].iloc[0] if len(t) else "none")):
        overlap = result.index.intersection(table.index)
        # Only fill where we still know nothing; earlier (more trusted) sources stand.
        fillable = overlap[result.loc[overlap, "label"] == UNKNOWN]
        result.loc[fillable, "label"] = table.loc[fillable, "label"].to_numpy()
        result.loc[fillable, "label_source"] = table.loc[fillable, "label_source"].to_numpy()
    result["label"] = result["label"].astype(int)
    return LabelTable(result)


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _coerce_label(value: object) -> int:
    """Map whatever the file says into 1 / 0 / -1."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN
    text = str(value).strip().lower()
    if text in ("1", "yes", "y", "true", "positive", "cancer"):
        return POSITIVE
    if text in ("0", "no", "n", "false", "negative"):
        return NEGATIVE
    if text in ("", "na", "nan", "null", "unknown", "."):
        return UNKNOWN
    # CDAS encodes the study year of diagnosis (0/1/2/…) in some extracts;
    # any parseable year means a cancer was confirmed.
    try:
        return POSITIVE if float(text) >= 0 else UNKNOWN
    except ValueError:
        return UNKNOWN


def _infer_source(path: Path) -> str:
    name = path.stem.lower()
    for source in SOURCE_PRIORITY:
        if source in name:
            return source
    return "manual"


def _source_rank(source: str) -> int:
    try:
        return SOURCE_PRIORITY.index(str(source))
    except ValueError:
        return len(SOURCE_PRIORITY)
