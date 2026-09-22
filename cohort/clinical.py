"""NLST outcomes and participant data — available without downloading any imaging.

**This supersedes CONTEXT.md §1.2's central assumption.** The brief states that
the authoritative per-patient outcome lives in the CDAS Participant table, behind
a formal application with a turnaround "measured in weeks", and builds a whole
two-track plan around waiting for it. That is no longer true: IDC ships the NLST
clinical tables alongside the imaging index, and `idc-index` will fetch them in
about two seconds.

What arrives:

    nlst_prsn    53,452 rows, 41 cols   the participant table — the one §1.2 says
                                        needs an application. Diagnosis timing,
                                        stage, histology, lesion size and
                                        location, demographics, smoking, and the
                                        official result of each screening round.
    nlst_canc     2,150 rows, 36 cols   one row per confirmed lung cancer
    nlst_screen  75,138 rows, 22 cols   per-round screening results
    nlst_ctab   177,487 rows, 14 cols   per-abnormality findings read off each CT
    nlst_ctabc   31,046 rows, 12 cols   abnormality comparisons between rounds

Every one of our 26,235 imaging patients joins to `nlst_prsn` on `pid`, so the
label column is complete: 1,059 positives and 25,176 negatives, both real rather
than assumed.

**A correction to §1.3 falls out of this.** The brief says positives are
"~2,050 / ~26,254 ≈ 8%" and calls that "the one number that dictates ML choices".
The 2,050 is right but it counts the *whole trial* — 2,058 patients across both
the CT and chest-X-ray arms of 53,452. Our imaging cohort is the CT arm alone, so
the true rate is 1,059 / 26,235 = **4.04%**, half what the brief assumes. The
imbalance is worse than planned for, not better, which makes AUPRC and
positive-oversampling more important rather than less.

Values in these tables are coded integers. The accompanying data dictionary maps
them to text, and `decode` uses it, so nothing here hard-codes that 1 means male
or that 110 means Stage IA.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

#: Tables IDC publishes for NLST.
TABLES = ("nlst_prsn", "nlst_canc", "nlst_screen", "nlst_ctab", "nlst_ctabc")

#: The participant table: one row per enrolled participant, both trial arms.
PARTICIPANTS = "nlst_prsn"

#: Markers the NLST extract uses in place of a value. `.N` is "not applicable"
#: — which for `candx_days` is how a *negative* is expressed, not missing data.
MISSING_CODES = {"", ".", ".M", ".N", ".F", ".A", "nan", "None"}

#: Screening rounds, and the per-round columns that describe them.
ROUND_COLUMNS = {
    "T0": {"result": "scr_res0", "days": "scr_days0", "isolation": "scr_iso0"},
    "T1": {"result": "scr_res1", "days": "scr_days1", "isolation": "scr_iso1"},
    "T2": {"result": "scr_res2", "days": "scr_days2", "isolation": "scr_iso2"},
}

#: ICD-O-3 morphology codes, which `de_type` carries as bare numbers because the
#: NLST dictionary ships no mapping for them. These are the 40 codes that
#: actually occur in the trial; anything else falls through as its raw code.
HISTOLOGY = {
    "8000": "Malignant neoplasm, NOS",
    "8001": "Malignant tumour cells",
    "8010": "Carcinoma, NOS",
    "8012": "Large cell carcinoma",
    "8013": "Large cell neuroendocrine carcinoma",
    "8021": "Carcinoma, anaplastic",
    "8022": "Pleomorphic carcinoma",
    "8032": "Spindle cell carcinoma",
    "8033": "Pseudosarcomatous carcinoma",
    "8041": "Small cell carcinoma",
    "8042": "Oat cell carcinoma",
    "8044": "Small cell carcinoma, intermediate cell",
    "8045": "Combined small cell carcinoma",
    "8046": "Non-small cell carcinoma",
    "8050": "Papillary carcinoma",
    "8052": "Papillary squamous cell carcinoma",
    "8070": "Squamous cell carcinoma",
    "8071": "Squamous cell carcinoma, keratinising",
    "8072": "Squamous cell carcinoma, large cell non-keratinising",
    "8075": "Squamous cell carcinoma, adenoid",
    "8083": "Basaloid squamous cell carcinoma",
    "8084": "Squamous cell carcinoma, clear cell type",
    "8140": "Adenocarcinoma, NOS",
    "8240": "Carcinoid tumour",
    "8246": "Neuroendocrine carcinoma",
    "8249": "Atypical carcinoid tumour",
    "8250": "Bronchioloalveolar adenocarcinoma",
    "8252": "Bronchioloalveolar carcinoma, non-mucinous",
    "8253": "Bronchioloalveolar carcinoma, mucinous",
    "8254": "Bronchioloalveolar carcinoma, mixed",
    "8255": "Adenocarcinoma with mixed subtypes",
    "8260": "Papillary adenocarcinoma",
    "8310": "Clear cell adenocarcinoma",
    "8323": "Mixed cell adenocarcinoma",
    "8430": "Mucoepidermoid carcinoma",
    "8480": "Mucinous adenocarcinoma",
    "8481": "Mucin-producing adenocarcinoma",
    "8490": "Signet ring cell carcinoma",
    "8550": "Acinar cell carcinoma",
    "8560": "Adenosquamous carcinoma",
    "8570": "Adenocarcinoma with squamous metaplasia",
    "8980": "Carcinosarcoma",
}

#: Boolean location flags in `nlst_prsn`, mapped to readable anatomy.
LESION_LOCATIONS = {
    "loclup": "left upper lobe",
    "locllow": "left lower lobe",
    "loclhil": "left hilum",
    "locrup": "right upper lobe",
    "locrmid": "right middle lobe",
    "locrlow": "right lower lobe",
    "locrhil": "right hilum",
    "locmed": "mediastinum",
    "loclmsb": "left main bronchus",
    "locrmsb": "right main bronchus",
    "loccar": "carina",
    "loclin": "lingula",
    "locoth": "other",
    "locunk": "unknown",
}


class ClinicalDataUnavailable(RuntimeError):
    """Raised when the NLST clinical tables are not on disk and cannot be fetched."""


def clinical_dir(fetch: bool = True) -> Path:
    """Locate IDC's clinical tables, fetching them if they are not present.

    The download is a few MB and takes a couple of seconds — nothing like the
    imaging — so fetching on demand is reasonable rather than a setup step.
    """
    from idc_index import IDCClient

    client = IDCClient()
    directory = Path(client.idc_data_dir) / "clinical_data"
    if (directory / PARTICIPANTS).is_dir():
        return directory

    if not fetch:
        raise ClinicalDataUnavailable(f"No clinical tables under {directory}")

    client.fetch_index("clinical_index")
    directory = Path(client.idc_data_dir) / "clinical_data"
    if not (directory / PARTICIPANTS).is_dir():
        raise ClinicalDataUnavailable(
            f"Fetched the clinical index but {PARTICIPANTS} is still missing from {directory}"
        )
    return directory


@functools.lru_cache(maxsize=8)
def load_table(name: str) -> pd.DataFrame:
    """Read one NLST clinical table, cached for the process."""
    if name not in TABLES:
        raise ValueError(f"Unknown NLST clinical table {name!r}; expected one of {TABLES}")

    directory = clinical_dir() / name
    parts = sorted(directory.glob("*.parquet"))
    if not parts:
        raise ClinicalDataUnavailable(f"No parquet files in {directory}")

    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if "pid" in frame:
        frame["pid"] = frame["pid"].astype(str).str.strip()
    return frame


@functools.lru_cache(maxsize=1)
def data_dictionary() -> pd.DataFrame:
    """Column labels and code-to-text mappings for the NLST tables.

    A fresh `IDCClient` always starts with `clinical_index = None`, and calling
    `fetch_index` to populate it re-runs the whole clinical download — a couple
    of seconds and a burst of log lines on every process start. Reading the
    installed parquet directly skips that; the fetch is only the fallback.
    """
    from idc_index import IDCClient

    client = IDCClient()
    installed = client.indices_overview.get("clinical_index", {})
    path = installed.get("file_path")

    index = None
    if installed.get("installed") and path and Path(path).exists():
        index = pd.read_parquet(path)
    else:
        client.fetch_index("clinical_index")
        index = client.clinical_index

    if index is None:
        raise ClinicalDataUnavailable("IDC clinical index could not be loaded")
    return index[index["collection_id"] == "nlst"]


@functools.lru_cache(maxsize=1)
def _code_maps() -> dict[tuple[str, str], dict[str, str]]:
    """{(table, column): {code: description}} built from the data dictionary."""
    maps: dict[tuple[str, str], dict[str, str]] = {}
    for row in data_dictionary().itertuples():
        options = getattr(row, "values", None)
        if options is None or len(options) == 0:
            continue
        mapping = {}
        for option in options:
            code = str(option.get("option_code", "")).strip()
            text = option.get("option_description")
            if code and text:
                mapping[code] = str(text).strip()
        if mapping:
            maps[(row.short_table_name, row.column)] = mapping
    return maps


#: Labels the dictionary spells out at length. Short forms read better in a
#: two-column panel, where a long label forces the value to wrap.
LABEL_OVERRIDES = {
    "age": "Age at randomisation",
    "de_type": "Histology",
    "de_stag": "Stage (AJCC 6)",
    "de_stag_7thed": "Stage (AJCC 7)",
    "de_grade": "Grade",
    "lesionsize": "Lesion size",
    "cigsmok": "Smoking at T0",
}


@functools.lru_cache(maxsize=1)
def _labels() -> dict[tuple[str, str], str]:
    """{(table, column): human-readable column label}."""
    out = {}
    for row in data_dictionary().itertuples():
        label = str(getattr(row, "column_label", "") or "").strip()
        if label:
            # Labels are often "short: long"; the short form reads better in a UI.
            out[(row.short_table_name, row.column)] = label.split(":")[0].strip()
    return out


def column_units(column: str) -> str | None:
    """Units to append to a value, where the short label drops them."""
    return {"age": "years", "lesionsize": "mm"}.get(column)


def is_missing(value) -> bool:
    """Whether a cell carries no value, in NLST's coding."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip() in MISSING_CODES


def decode(table: str, column: str, value) -> str | None:
    """Turn a coded value into its documented description.

    Falls back to the raw value when the dictionary has no mapping, and to None
    when the cell is empty — so a caller can always render the result directly.
    """
    if is_missing(value):
        return None
    text = str(value).strip()
    if column in ("de_type", "de_type_7thed") and text in HISTOLOGY:
        return f"{HISTOLOGY[text]} (ICD-O-3 {text})"
    return _code_maps().get((table, column), {}).get(text, text)


def column_label(table: str, column: str) -> str:
    if column in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[column]
    return _labels().get((table, column), column)


def as_int(value) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ the label


def outcomes() -> pd.DataFrame:
    """One row per participant: the authoritative label plus its supporting facts.

    `label` is 1 when a lung cancer diagnosis is on record and 0 otherwise. The
    0 is a real negative, not an assumption: `nlst_prsn` covers every enrolled
    participant and `canc_free_days` records how long they were followed without
    one. That is the column `ml/train.py` and `ml/evaluate.py` have been waiting
    for, and it is what makes `-1 = unknown` the exception rather than the rule.
    """
    people = load_table(PARTICIPANTS)
    diagnosed = ~people["candx_days"].map(is_missing)

    frame = pd.DataFrame({
        "PatientID": people["pid"],
        "label": diagnosed.astype(int),
        "label_source": "idc",
        "days_to_diagnosis": people["candx_days"].map(as_int),
        "cancer_free_days": people["canc_free_days"].map(as_int),
        "diagnosis_study_year": people["cancyr"].map(as_int),
    })
    return frame.drop_duplicates(subset="PatientID").reset_index(drop=True)


def label_file(path: Path) -> Path:
    """Write the outcomes to a CSV that `cohort/make_splits.py --labels` accepts."""
    frame = outcomes()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# -------------------------------------------------------- per-patient record


def patient_record(patient_id: str) -> dict:
    """Everything the clinical tables know about one patient.

    Structured for display: each section is a list of {label, value} pairs
    already decoded to text, so the viewer renders them without needing its own
    copy of the NLST codebook.
    """
    people = load_table(PARTICIPANTS)
    rows = people[people["pid"] == str(patient_id)]
    if rows.empty:
        return {"patient_id": str(patient_id), "found": False}
    row = rows.iloc[0]

    def field(column: str) -> dict:
        value = decode(PARTICIPANTS, column, row.get(column))
        units = column_units(column)
        if value is not None and units and units not in value:
            value = f"{value} {units}"
        return {
            "column": column,
            "label": column_label(PARTICIPANTS, column),
            "value": value,
        }

    diagnosed = not is_missing(row.get("candx_days"))
    days = as_int(row.get("candx_days"))
    free_days = as_int(row.get("cancer_free_days") if "cancer_free_days" in row else row.get("canc_free_days"))

    record: dict = {
        "patient_id": str(patient_id),
        "found": True,
        "label": int(diagnosed),
        "label_source": "idc",
        "outcome": {
            "diagnosed": diagnosed,
            "days_to_diagnosis": days,
            "years_to_diagnosis": round(days / 365.25, 2) if days is not None else None,
            "diagnosis_study_year": decode(PARTICIPANTS, "cancyr", row.get("cancyr")),
            "cancer_free_days": free_days,
            "screen_result_at_diagnosis": decode(PARTICIPANTS, "can_scr", row.get("can_scr")),
        },
        "demographics": [field(c) for c in ("age", "gender", "race", "cigsmok")],
        "screening": [],
        "tumour": [],
        "locations": [],
    }

    for round_label, columns in ROUND_COLUMNS.items():
        result = decode(PARTICIPANTS, columns["result"], row.get(columns["result"]))
        if result is None and is_missing(row.get(columns["days"])):
            continue
        record["screening"].append({
            "round": round_label,
            "result": result,
            "days_from_randomisation": as_int(row.get(columns["days"])),
        })

    if diagnosed:
        for column in ("de_stag", "de_stag_7thed", "de_type", "de_grade", "lesionsize"):
            entry = field(column)
            if entry["value"] is not None:
                record["tumour"].append(entry)
        record["locations"] = [
            name for column, name in LESION_LOCATIONS.items()
            if str(row.get(column, "")).strip() in ("1", "1.0", "True")
        ]

    return record


def cohort_summary(patient_ids: list[str] | None = None) -> dict:
    """Outcome counts over the cohort, for reporting and sanity checks."""
    frame = outcomes()
    if patient_ids is not None:
        frame = frame[frame["PatientID"].isin({str(p) for p in patient_ids})]

    positives = int((frame["label"] == 1).sum())
    return {
        "patients": int(len(frame)),
        "positive": positives,
        "negative": int((frame["label"] == 0).sum()),
        "positive_rate": round(positives / len(frame), 4) if len(frame) else None,
        "by_diagnosis_year": (
            frame.loc[frame["label"] == 1, "diagnosis_study_year"]
            .value_counts().sort_index().to_dict()
        ),
        "median_follow_up_days": int(frame["cancer_free_days"].median())
        if frame["cancer_free_days"].notna().any() else None,
    }


# ------------------------------------------------------------------------- CLI


def format_record(record: dict) -> str:
    """Render a patient record for a terminal."""
    if not record.get("found"):
        return f"Patient {record['patient_id']}: no NLST participant record."

    outcome = record["outcome"]
    lines = [f"Patient {record['patient_id']}"]

    if outcome["diagnosed"]:
        when = f"study year {outcome['diagnosis_study_year']}" if outcome["diagnosis_study_year"] else ""
        years = f"{outcome['years_to_diagnosis']} years from randomisation" if outcome["years_to_diagnosis"] else ""
        lines.append(f"  OUTCOME     lung cancer diagnosed — {when}, {years}".rstrip(", "))
        if outcome["screen_result_at_diagnosis"]:
            lines.append(f"              screen at diagnosis: {outcome['screen_result_at_diagnosis']}")
    else:
        followed = outcome["cancer_free_days"]
        span = f" over {followed / 365.25:.1f} years of follow-up" if followed else ""
        lines.append(f"  OUTCOME     no lung-cancer diagnosis on record{span}")

    for section, title in (("tumour", "TUMOUR"), ("demographics", "PERSON")):
        entries = [e for e in record.get(section, []) if e.get("value")]
        if not entries:
            continue
        lines.append(f"  {title:<11} " + " · ".join(f"{e['label']}: {e['value']}" for e in entries))

    if record.get("locations"):
        lines.append(f"  LOCATION    {', '.join(record['locations'])}")

    for entry in record.get("screening", []):
        day = entry["days_from_randomisation"]
        lines.append(
            f"  {entry['round']:<11} day {day if day is not None else '?':>4}  {entry['result'] or '—'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="NLST outcomes, available without downloading any imaging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--patient", help="show everything known about one patient")
    p.add_argument("--summary", action="store_true", help="outcome counts over the whole cohort")
    p.add_argument("--export", type=Path, default=None, help="write labels to a CSV")
    args = p.parse_args(argv)

    if args.patient:
        print(format_record(patient_record(args.patient)))
        return 0

    if args.export:
        print(f"Wrote {label_file(args.export)}")
        return 0

    import json

    print(json.dumps(cohort_summary(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
