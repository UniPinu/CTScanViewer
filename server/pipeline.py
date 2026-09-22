"""Orchestration: the stages, run in order, for one patient.

`server/app.py` owns HTTP and `server/jobs.py` owns threading; this module owns
the actual sequence — fetch, preprocess, infer, assemble — and is deliberately
free of both, so it can be driven from a notebook, a CLI or a test without a
server running.

It is also where the **results object** (§5.3) is assembled. That object is the
contract with the viewer, and the one place allowed to decide what the UI is
told. Two rules it enforces:

  * A stage that did not run says so explicitly. The risk block carries
    `unavailable: true` and a reason rather than being absent or, worse, zero.
  * The summary is rendered from the assembled object, never from intermediate
    state, so the prose and the numbers on screen cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cohort.split import assign_split
from core import jsonio, paths
from core.provenance import stamp
from ingest.stream import SeriesCache
from ml import saliency, summary
from ml.change import compare, load_interior
from ml.risk import RiskUnavailable, cohort_percentile, default_calibrator, load_backend
from preprocess.export_volumes import export_series, write_index
from preprocess.volume import load_series

#: Tasks `/infer` understands.
TASKS = ("risk", "change", "saliency")


@dataclass
class Reporter:
    """How the pipeline narrates itself. The server wires this to a Job."""

    on_stage: object = None  # Callable[[str, str, float | None], None]

    def __call__(self, stage: str, message: str = "", progress: float | None = None) -> None:
        if self.on_stage:
            self.on_stage(stage, message, progress)  # type: ignore[operator]


def cohort_table() -> pd.DataFrame:
    if not paths.SERIES_PARQUET.exists():
        raise FileNotFoundError(
            f"No cohort at {paths.SERIES_PARQUET}. Run `python -m cohort.build_cohort` "
            "and `python -m cohort.make_splits` first."
        )
    return pd.read_parquet(paths.SERIES_PARQUET)


def splits_table() -> pd.DataFrame | None:
    if not paths.SPLITS_PARQUET.exists():
        return None
    return pd.read_parquet(paths.SPLITS_PARQUET)


def patient_series(patient_id: str, primary_only: bool = True) -> pd.DataFrame:
    """A patient's series, one per round when `primary_only`."""
    cohort = cohort_table()
    mine = cohort[cohort["PatientID"].astype(str) == str(patient_id)]
    if primary_only and "is_primary" in mine:
        mine = mine[mine["is_primary"]]
    return mine.sort_values("round_key")


def patient_overview(patient_id: str) -> dict:
    """Everything known about a patient *before* any imaging is fetched.

    Two halves, both answerable from metadata alone:

      **What is stored** — how many screening rounds exist, which
      reconstructions, their slice thickness and size, and whether each is
      already in the local cache. All of it from `series.parquet`, which was
      built by a SQL query against IDC's index.

      **What the outcome was** — the NLST participant record: whether this
      person was diagnosed, when, at what stage and histology, plus the official
      result of each screening round. From IDC's clinical tables
      (`cohort/clinical.py`), also without downloading imaging.

    Backs `GET /patient/{id}`, so the viewer can show a full picture of a case
    and let someone decide whether the download is worth it.
    """
    series = patient_series(patient_id, primary_only=False)
    if series.empty:
        raise KeyError(f"Patient {patient_id} is not in the cohort")

    cache = SeriesCache()
    primary = series[series["is_primary"]] if "is_primary" in series else series
    clinical_record = _clinical_record(patient_id)
    screening = {s["round"]: s for s in clinical_record.get("screening", [])}

    rounds = []
    for _, row in primary.iterrows():
        uid = str(row["SeriesInstanceUID"])
        label = row.get("round") or row.get("round_key")
        # Reconstructions of this round that we are *not* using as primary.
        alternates = series[(series["round_key"] == row.get("round_key")) & (~series["is_primary"])]
        rounds.append({
            "round": label,
            "study_date": str(row.get("StudyDate", "")),
            "series_uid": uid,
            "kernel": row.get("kernel"),
            "kernel_style": row.get("kernel_style"),
            "slice_mm": None if pd.isna(row.get("slice_mm")) else float(row.get("slice_mm")),
            "instances": int(row.get("instanceCount", 0)),
            "size_mb": round(float(row.get("series_size_MB", 0.0)), 1),
            "cached": cache.contains(uid),
            "manufacturer": str(row.get("Manufacturer", "") or ""),
            "model": str(row.get("ManufacturerModelName", "") or ""),
            "alternate_reconstructions": int(len(alternates)),
            # The official NLST read for this round, beside the scan itself.
            "screen_result": (screening.get(label) or {}).get("result"),
            "screen_days": (screening.get(label) or {}).get("days_from_randomisation"),
        })

    cached_mb = sum(r["size_mb"] for r in rounds if r["cached"])
    return {
        "patient_id": str(patient_id),
        "split": assign_split(str(patient_id)),
        "n_rounds": len(rounds),
        "n_series": int(len(series)),
        "total_mb": round(float(series["series_size_MB"].sum()), 1),
        "primary_mb": round(float(primary["series_size_MB"].sum()), 1),
        "cached_mb": round(cached_mb, 1),
        "rounds": rounds,
        "clinical": clinical_record,
        "label": clinical_record.get("label", -1),
        "label_source": clinical_record.get("label_source", "none"),
        "has_results": results_path(patient_id).exists(),
        "exported": exported_patient() == str(patient_id),
    }


def _clinical_record(patient_id: str) -> dict:
    """The NLST participant record, or an explicit 'not available' marker.

    Clinical data is a separate IDC download, so it can legitimately be absent.
    That is reported rather than raised: a missing outcome should not stop
    someone browsing the imaging.
    """
    try:
        from cohort import clinical

        return clinical.patient_record(str(patient_id))
    except Exception as e:  # noqa: BLE001 - absent clinical data is not fatal
        return {
            "patient_id": str(patient_id),
            "found": False,
            "label": -1,
            "label_source": "none",
            "error": f"{type(e).__name__}: {e}",
        }


def cohort_listing(split: str | None = None, limit: int = 50, min_rounds: int = 1,
                   query: str | None = None, label: int | None = None) -> list[dict]:
    """Patients for the left rail — backs `GET /cohort`."""
    splits = splits_table()
    if splits is None:
        cohort = cohort_table()
        primary = cohort[cohort["is_primary"]] if "is_primary" in cohort else cohort
        counts = primary.groupby(primary["PatientID"].astype(str))["round_key"].nunique()
        splits = pd.DataFrame({
            "PatientID": counts.index,
            "n_rounds": counts.to_numpy(),
            "label": -1,
            "label_source": "none",
        })
        splits["split"] = [assign_split(p) for p in splits["PatientID"]]

    frame = splits[splits["n_rounds"] >= min_rounds]
    if split:
        frame = frame[frame["split"] == split]
    if query:
        frame = frame[frame["PatientID"].astype(str).str.contains(str(query), na=False)]
    if label is not None:
        frame = frame[frame["label"] == int(label)]

    return [
        {
            "patient_id": str(row["PatientID"]),
            "split": row["split"],
            "n_rounds": int(row["n_rounds"]),
            "label": int(row.get("label", -1)),
            "label_source": row.get("label_source", "none"),
        }
        for _, row in frame.head(limit).iterrows()
    ]


def random_patient(split: str | None = "test", min_rounds: int = 2, seed: int | None = None) -> str:
    """Pick a patient at random — backs `GET /random-patient`.

    Defaults to the **test** split and to patients with at least two rounds, so
    the button lands on someone the model never trained on and for whom change
    detection has something to compare.
    """
    import random

    candidates = cohort_listing(split=split, limit=1_000_000, min_rounds=min_rounds)
    if not candidates:
        raise LookupError(f"No patient with >= {min_rounds} rounds in split {split!r}")
    return random.Random(seed).choice(candidates)["patient_id"]


# ------------------------------------------------------------------- the stages


def exported_patient() -> str | None:
    """Which patient currently occupies `viewer/public/data`."""
    marker = paths.VIEWER_DATA / "patient.json"
    if not marker.exists():
        return None
    try:
        return str(jsonio.read(marker).get("patient_id"))
    except Exception:
        return None


def results_path(patient_id: str) -> Path:
    return paths.RESULTS_DIR / f"{patient_id}.json"


def fetch_patient(patient_id: str, report: Reporter, cache: SeriesCache | None = None,
                  with_masks: bool = True) -> dict:
    """Stage 1+2 — stream the patient's series and export them for the viewer."""
    import shutil

    series = patient_series(patient_id, primary_only=True)
    if series.empty:
        raise KeyError(f"Patient {patient_id} has no primary series")

    cache = cache or SeriesCache()
    report("clearing", "Clearing the previous patient's volumes", None)
    if paths.VIEWER_DATA.exists():
        shutil.rmtree(paths.VIEWER_DATA, ignore_errors=True)
    paths.VIEWER_DATA.mkdir(parents=True, exist_ok=True)

    total_mb = float(series["series_size_MB"].sum())
    entries: list[dict] = []
    done_mb = 0.0

    # Export each series while it is still resident: the cache may evict it to
    # make room for the next one (see SeriesCache.stream).
    for i, (directory, (_, row)) in enumerate(zip(cache.stream(series), series.iterrows()), 1):
        done_mb += float(row["series_size_MB"])
        report("download", f"Downloaded {done_mb:.0f} of {total_mb:.0f} MB", done_mb / max(total_mb, 1))
        report("preprocess", f"Preparing round {i} of {len(series)}", done_mb / max(total_mb, 1))
        entries.append(export_series(directory, paths.VIEWER_DATA, with_mask=with_masks))

    write_index(paths.VIEWER_DATA, entries)
    jsonio.write(paths.VIEWER_DATA / "patient.json", {
        "patient_id": str(patient_id),
        "split": assign_split(str(patient_id)),
        "rounds": sorted({e["round"] for e in entries if e.get("round")}),
        "n_series": len(entries),
    })
    return {"patient_id": str(patient_id), "series": len(entries), "mb": round(total_mb)}


def run_risk(patient_id: str, series: pd.DataFrame, report: Reporter,
             cache: SeriesCache, want_saliency: bool) -> tuple[dict, dict | None]:
    """Stage 3a+3c — risk and attention for the most recent round.

    Risk is predicted from the *latest* scan, which is the one a screening
    decision would actually be made on.
    """
    latest = series.iloc[-1]
    backend = load_backend()
    report("risk", f"Running {backend.name} on round {latest.get('round_key')}", None)

    attention_dir = paths.RESULTS_DIR / str(patient_id) / "attention" if want_saliency else None
    try:
        directory = cache.get(
            str(latest["SeriesInstanceUID"]),
            expected_instances=int(latest["instanceCount"]),
            expected_mb=float(latest["series_size_MB"]),
        )
        prediction = backend.predict(directory, attention_dir)
    except RiskUnavailable as e:
        # A first-class state, not an error: the rest of the report is still valid.
        return {"unavailable": True, "reason": str(e), "model": "none"}, None

    calibrator = default_calibrator()
    scores = calibrator.apply(prediction.scores)
    prediction.scores = scores
    prediction.calibrated = calibrator.fitted

    risk = prediction.as_dict()
    risk["round"] = latest.get("round_key")
    risk["model"] = prediction.model
    risk["cohort_percentile"] = cohort_percentile(scores[0], reference_distribution())

    attention = None
    if want_saliency and prediction.attention_dir:
        report("saliency", "Rendering attention overlay", None)
        attention = build_attention_overlay(patient_id, prediction.attention_dir, latest, cache)
    return risk, attention


def build_attention_overlay(patient_id: str, attention_dir: Path, latest, cache: SeriesCache) -> dict | None:
    """Put Sybil's attention onto the viewer's grid and save it."""
    directory = cache.path_of(str(latest["SeriesInstanceUID"]))
    if not directory.exists():
        return None
    series = load_series(directory)
    overlay = saliency.attention_overlay(attention_dir, series.hu.shape)
    if overlay is None:
        return None

    # Risk runs on the latest round, so the attention describes that round only.
    overlay.round = latest.get("round_key")
    overlay.patient_id = str(patient_id)
    out_dir = paths.VIEWER_DATA / "overlays" / "attention"
    meta = overlay.save(out_dir)
    meta["overlay_dir"] = "overlays/attention"
    return meta


def run_change(patient_id: str, series: pd.DataFrame, report: Reporter,
               cache: SeriesCache) -> dict:
    """Stage 3b — compare consecutive rounds.

    Both rounds of a pair must be on disk at once, so this asks the cache for
    them by path rather than streaming, and checks the budget first.
    """
    if len(series) < 2:
        return {"pairs": [], "note": "Only one screening round is available, so there is "
                                     "nothing to compare."}

    pairs = []
    for earlier_row, later_row in zip(series.iloc[:-1].itertuples(), series.iloc[1:].itertuples()):
        from_round = getattr(earlier_row, "round_key", "earlier")
        to_round = getattr(later_row, "round_key", "later")
        report("change", f"Registering {from_round} to {to_round}", None)

        earlier_dir = cache.get(str(earlier_row.SeriesInstanceUID),
                                expected_instances=int(earlier_row.instanceCount),
                                expected_mb=float(earlier_row.series_size_MB))
        later_dir = cache.get(str(later_row.SeriesInstanceUID),
                              expected_instances=int(later_row.instanceCount),
                              expected_mb=float(later_row.series_size_MB))
        if not earlier_dir.exists():
            # The later fetch evicted the earlier series: the cache cap is too
            # small for a pair. Say so rather than failing obscurely.
            raise RuntimeError(
                f"The cache evicted round {from_round} while fetching {to_round}. "
                f"Raise the cap (CT_CACHE_GB) to at least "
                f"{(earlier_row.series_size_MB + later_row.series_size_MB) / 1000:.1f} GB."
            )

        out_dir = paths.RESULTS_DIR / str(patient_id) / f"change_{from_round}_{to_round}"
        result = compare(
            load_series(earlier_dir), load_series(later_dir),
            from_round=from_round, to_round=to_round, out_dir=out_dir,
        )

        entry = result.as_dict()
        entry["overlay_dir"] = write_change_overlay(patient_id, result, out_dir, from_round, to_round)
        pairs.append(entry)

    return {"pairs": pairs}


def write_change_overlay(patient_id: str, result, out_dir: Path,
                         from_round: str, to_round: str) -> str | None:
    """Render the change map onto the viewer's grid for the earlier round."""
    delta = np.load(out_dir / "delta.npy")
    # Confine the heatmap to the voxels the regions were found in, so the
    # overlay and the region list describe the same thing.
    interior = load_interior(out_dir, delta.shape)
    overlay = saliency.change_overlay(delta, mask=interior)
    # A change map lives in the geometry of the earlier round of the pair.
    overlay.round = from_round
    overlay.patient_id = str(patient_id)

    # The viewer shows the original series geometry; change ran at 1.5 mm.
    index_path = paths.VIEWER_DATA / "index.json"
    if index_path.exists():
        entries = jsonio.read(index_path)
        target = next((e for e in entries if e.get("round") == from_round), None)
        if target:
            overlay = saliency.resample_overlay_to(overlay, tuple(target["shape"]))

    relative = f"overlays/change_{from_round}_{to_round}"
    overlay.save(paths.VIEWER_DATA / relative)
    return relative


def reference_distribution() -> np.ndarray | None:
    """Cohort-wide 1-year scores, for the percentile in the summary.

    None until a batch run has produced one; the percentile is then omitted
    rather than invented (see `ml/risk.cohort_percentile`).
    """
    path = paths.REPORTS_DIR / "risk_reference.npy"
    return np.load(path) if path.exists() else None


def infer(patient_id: str, tasks: list[str], report: Reporter) -> dict:
    """Run the requested tasks and assemble the results object (§5.3)."""
    series = patient_series(patient_id, primary_only=True)
    if series.empty:
        raise KeyError(f"Patient {patient_id} has no primary series")

    # Overlays are written into `viewer/public/data` and are resampled onto the
    # geometry recorded in the index.json already sitting there. If that data
    # belongs to a different patient, the overlay is silently fitted to *their*
    # scan and then passes every downstream check — one patient's findings drawn
    # over another's anatomy. Exporting first makes the directory consistent by
    # construction rather than by the caller remembering to fetch.
    if exported_patient() != str(patient_id):
        report("preprocess", f"Exporting patient {patient_id} for the viewer", None)
        fetch_patient(patient_id, report)

    cache = SeriesCache()
    results: dict = {
        "patient_id": str(patient_id),
        "split": assign_split(str(patient_id)),
        "rounds": [str(r) for r in series["round_key"].tolist()],
        "tasks": tasks,
    }

    attention = None
    if "risk" in tasks:
        risk, attention = run_risk(patient_id, series, report, cache, "saliency" in tasks)
        results["risk"] = risk
    if attention:
        results["attention"] = attention
    if "change" in tasks:
        results["change"] = run_change(patient_id, series, report, cache)

    model_name = (results.get("risk") or {}).get("model", "none")
    results["provenance"] = stamp(model_name, tasks=tasks)
    # Rendered last, from the finished object, so the prose cannot disagree with
    # the numbers beside it.
    results["summary"] = summary.compose(results)

    jsonio.write(results_path(patient_id), results)
    report("done", "Results ready", 1.0)
    return results
