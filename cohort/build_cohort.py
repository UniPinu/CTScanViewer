"""Stage 0a — build the master list of usable NLST CT series (CONTEXT.md §4).

Queries the IDC metadata index with SQL and writes one row per series to
`data/cohort/series.parquet`. Nothing is downloaded: `idc-index` ships the
~100 TB catalogue's *metadata* as a local DuckDB file, so this is a local query
against a few hundred MB, and the imaging stays in the public buckets until a
later stage actually asks for it (§2.2).

The frozen `manifest_*.s5cmd` in the repo root is a snapshot of roughly this
query. It remains as an offline fallback (`--from-manifest`), but the live query
is preferred so the cohort is rebuildable and versioned.

Usage:
    python -m cohort.build_cohort                    # full NLST CT cohort
    python -m cohort.build_cohort --limit 2000       # a quick slice, for development
    python -m cohort.build_cohort --from-manifest    # offline, from the frozen manifest
    python -m cohort.build_cohort --summary          # describe an existing series.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core import paths
from core.nlst import parse_series_description

#: Columns pulled from the IDC index. Everything else is derived locally.
INDEX_COLUMNS = [
    "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "StudyDate", "SeriesDescription",
    "Modality", "instanceCount", "series_size_MB", "crdc_series_uuid", "series_aws_url",
    "Manufacturer", "ManufacturerModelName",
]

#: Preference order for `kernel_style` when picking one series per study.
#: Alphabetical order would rank "other" above "sharp" and "smooth", which is
#: backwards, so the sort uses this explicit rank instead.
KERNEL_RANK = {"smooth": 0, "other": 1, "sharp": 2}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", default="nlst", help="IDC collection_id (default: nlst)")
    p.add_argument("--out", type=Path, default=paths.SERIES_PARQUET,
                   help=f"output parquet (default: {paths.SERIES_PARQUET})")
    p.add_argument("--min-slices", type=int, default=10,
                   help="drop series with fewer slices, i.e. scouts (default: 10)")
    p.add_argument("--limit", type=int, default=None, help="keep only the first N patients, for development")
    p.add_argument("--from-manifest", action="store_true",
                   help="use the frozen s5cmd manifest instead of querying IDC")
    p.add_argument("--manifest", type=Path, default=None, help="manifest to use with --from-manifest")
    p.add_argument("--summary", action="store_true", help="print a summary of the existing --out and exit")
    return p.parse_args(argv)


def query_idc(collection: str, min_slices: int) -> pd.DataFrame:
    """One row per CT series in the collection, straight from the local index."""
    from idc_index import IDCClient  # slow import; keep it out of --summary and --help

    client = IDCClient()
    columns = ", ".join(INDEX_COLUMNS)
    # Filtering scouts in SQL means we never materialise the 590k-row NLST slab
    # in pandas just to throw most of it away.
    return client.sql_query(
        f"SELECT {columns} FROM index "
        f"WHERE collection_id = '{collection}' AND Modality = 'CT' "
        f"AND instanceCount >= {int(min_slices)}"
    )


def read_manifest(manifest: Path, min_slices: int) -> pd.DataFrame:
    """Offline fallback: join the frozen s5cmd manifest against the local index."""
    from idc_index import IDCClient

    from ingest.download_sample import UUID_RE, split_manifest

    _, cp_lines = split_manifest(manifest)
    uuids = pd.DataFrame({
        "crdc_series_uuid": [UUID_RE.search(line).group(1) for line in cp_lines],
    })
    client = IDCClient()
    df = uuids.merge(client.index[INDEX_COLUMNS], on="crdc_series_uuid", how="inner")
    return df[(df["Modality"] == "CT") & (df["instanceCount"] >= min_slices)]


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Decode SeriesDescription into the columns the rest of the pipeline uses."""
    described = df["SeriesDescription"].map(parse_series_description)
    out = df.copy()
    out["PatientID"] = out["PatientID"].astype(str)
    out["study_year"] = [d.study_year for d in described]
    out["round"] = [d.round_label for d in described]
    out["kernel"] = [d.kernel for d in described]
    out["kernel_style"] = [d.kernel_style for d in described]
    out["slice_mm"] = [d.slice_mm for d in described]
    out["is_axial"] = [d.is_axial for d in described]
    out["round_key"] = round_key(out)
    return out


def round_key(df: pd.DataFrame) -> pd.Series:
    """The screening round a series belongs to — the pipeline's real unit of time.

    Usually just T0/T1/T2 off the description. Two things force a fallback:
    descriptions that do not parse, and the handful of patients who have *two
    distinct studies inside one round* (a repeat acquisition, or two scanners on
    the same day — patient 123342 has both a Siemens and a Toshiba study at T2).
    Grouping by study would hand the change head two competing baselines for one
    round; grouping by round collapses them into one choice.
    """
    return df["round"].fillna("D" + df["StudyDate"].astype(str))


def select_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Mark exactly one series per study as the one models should read (§4 Stage 0).

    NLST reconstructs each acquisition several ways — typically a smooth
    STANDARD and a sharp LUNG/BONE rendering of identical anatomy. A model must
    see one of them, chosen the same way every time, or results are not
    reproducible. The choice is made per *screening round* rather than per
    study, so a patient scanned twice in one round still yields exactly one
    baseline for the change head to work from. Preference order:

        1. axial reconstructions over localizers
        2. smooth kernels over sharp ones (less noise; what Sybil expects)
        3. thinner slices (more z-resolution for small nodules)
        4. more slices (better coverage of the chest)
        5. a hash of the SeriesInstanceUID, purely to break remaining ties
           deterministically rather than by row order

    The losers stay in the table — the viewer still offers them, and the
    before/after view uses them to show what a reconstruction kernel actually does.
    """
    from cohort.split import split_value  # the same stable hash, reused as a tiebreak

    ranked = df.copy()
    ranked["_kernel_rank"] = ranked["kernel_style"].map(KERNEL_RANK).fillna(len(KERNEL_RANK))
    # slice_mm is None for series whose description omitted it; sort those last
    # rather than letting NaN land wherever the sort happens to put it.
    ranked["_slice_mm"] = ranked["slice_mm"].fillna(1e9)
    ranked["_tiebreak"] = ranked["SeriesInstanceUID"].map(
        lambda uid: split_value(str(uid), seed="series-pick")
    )
    ranked = ranked.sort_values(
        by=["PatientID", "round_key", "is_axial", "_kernel_rank",
            "_slice_mm", "instanceCount", "_tiebreak"],
        ascending=[True, True, False, True, True, False, True],
    )
    ranked["is_primary"] = ~ranked.duplicated(subset=["PatientID", "round_key"], keep="first")
    return ranked.drop(columns=["_kernel_rank", "_slice_mm", "_tiebreak"])


def summarise(df: pd.DataFrame) -> str:
    lines = [
        f"series            {len(df):,}",
        f"patients          {df['PatientID'].nunique():,}",
        f"studies           {df['StudyInstanceUID'].nunique():,}",
        f"total size        {df['series_size_MB'].sum() / 1e6:.2f} TB",
    ]
    if "is_primary" in df:
        primary = df[df["is_primary"]]
        rounds = primary.groupby("round").size().sort_index().to_dict()
        per_patient = primary.groupby("PatientID")["round_key"].nunique()
        lines.insert(1, f"  primary         {len(primary):,}")
        lines.append(f"rounds            {rounds}")
        lines.append(f"rounds/patient    {per_patient.value_counts().sort_index().to_dict()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.summary:
        if not args.out.exists():
            raise SystemExit(f"No cohort at {args.out} — run without --summary first")
        print(summarise(pd.read_parquet(args.out)))
        return 0

    if args.from_manifest:
        manifest = args.manifest or paths.default_manifest()
        if manifest is None or not manifest.exists():
            raise SystemExit("No manifest_*.s5cmd found; drop --from-manifest to query IDC instead")
        print(f"Reading frozen manifest {manifest.name} …", file=sys.stderr)
        df = read_manifest(manifest, args.min_slices)
    else:
        print(f"Querying the IDC index for collection '{args.collection}' …", file=sys.stderr)
        df = query_idc(args.collection, args.min_slices)

    if df.empty:
        raise SystemExit(f"No CT series found for collection '{args.collection}'")

    df = annotate(df)
    df = df[df["is_axial"]]  # scouts carry no anatomy worth modelling
    if df.empty:
        raise SystemExit("Every series was a localizer — check --collection")

    if args.limit:
        keep = sorted(df["PatientID"].unique())[: args.limit]
        df = df[df["PatientID"].isin(keep)]

    df = select_primary(df).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\nWrote {args.out}")
    print(summarise(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
