"""Stage 1 — fetch DICOM series on demand, under a hard disk budget (§4, §2.2).

The NLST imaging is ~11 TB. We never hold it. This module treats the public
buckets as the storage tier and the local disk as a **bounded LRU cache**: ask
for a series, get a directory of `.dcm` files back, and trust that the working
set stays under the cap no matter how long a training run iterates.

That is what makes "train on the train split" tractable without a download step:
the loop streams, the cache keeps a rolling window of recently-touched series,
and the least-recently-used ones are evicted when the next fetch would breach
the budget.

Design notes:

  * **The cache is disposable.** Every entry can be re-fetched from IDC, so
    eviction loses time, never information. Corrupt or half-written entries are
    discarded rather than repaired — a partial download that looked complete
    would poison every downstream stage silently.
  * **Completeness is checked, not assumed.** A download is only committed once
    the directory holds the instance count the index promised. Until then it
    lives under a `.partial` name that nothing else will read.
  * **The index is a cache, not a ledger.** `cache_index.json` speeds up
    bookkeeping, but the directories on disk are the truth; a lost or corrupt
    index is rebuilt by scanning.

Usage:
    python -m ingest.stream --series 1.2.840...        # fetch one series
    python -m ingest.stream --patient 100012           # fetch a patient's primary series
    python -m ingest.stream --status                   # what is cached, and how big
    python -m ingest.stream --evict-all                # empty the cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core import jsonio, paths

#: Bookkeeping lives beside the cached series, not inside any of them.
INDEX_NAME = "cache_index.json"

ProgressFn = Callable[[str, str], None]


@dataclass
class CacheEntry:
    series_uid: str
    bytes: int
    instances: int
    last_used: float
    fetched_at: float

    def as_dict(self) -> dict:
        return {
            "series_uid": self.series_uid,
            "bytes": self.bytes,
            "instances": self.instances,
            "last_used": self.last_used,
            "fetched_at": self.fetched_at,
        }


@dataclass
class SeriesCache:
    """An LRU-bounded directory of DICOM series, keyed by SeriesInstanceUID."""

    root: Path = field(default_factory=lambda: paths.CACHE_DIR)
    cap_bytes: int = field(default_factory=paths.cache_cap_bytes)
    on_progress: ProgressFn | None = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, CacheEntry] = self._load_index()

    # ---------------------------------------------------------------- public

    def path_of(self, series_uid: str) -> Path:
        """Where this series lives, cached or not. Does not fetch."""
        return self.root / series_uid

    def contains(self, series_uid: str) -> bool:
        return series_uid in self._entries and self.path_of(series_uid).is_dir()

    def get(self, series_uid: str, expected_instances: int | None = None,
            expected_mb: float | None = None) -> Path:
        """Return a directory of `.dcm` files for this series, fetching if needed.

        On a hit the entry is touched so it moves to the back of the eviction
        queue. On a miss we make room *first* — evicting after the fact would
        let the disk spike past the budget mid-download.
        """
        if self.contains(series_uid):
            self._touch(series_uid)
            return self.path_of(series_uid)

        incoming = int((expected_mb or 0) * 1e6)
        self.make_room(incoming)
        return self._fetch(series_uid, expected_instances)

    def stream(self, series: pd.DataFrame) -> Iterator[Path]:
        """Yield each series' directory in turn, fetching as it goes.

        This is a generator on purpose, and the distinction matters. A list of
        paths would be a lie whenever the request is larger than the cap: by the
        time the last series had been fetched, the first would already have been
        evicted and its path would point at nothing. Yielding forces the caller
        to finish with each series before the next fetch can reclaim its space,
        which is exactly the contract a bounded cache can actually honour.

        The consequence for callers: **use the path before asking for the next
        one.** Do not collect these into a list and process them afterwards.

        `series` needs `SeriesInstanceUID` and, ideally, `instanceCount` and
        `series_size_MB` from the cohort table.
        """
        for _, row in series.iterrows():
            yield self.get(
                str(row["SeriesInstanceUID"]),
                expected_instances=int(row["instanceCount"]) if "instanceCount" in row else None,
                expected_mb=float(row["series_size_MB"]) if "series_size_MB" in row else None,
            )

    def fits_together(self, series: pd.DataFrame) -> bool:
        """Whether these series can all be cached at once.

        The change head needs two rounds resident simultaneously to register one
        against the other, so it asks this first rather than discovering the
        problem as a vanished directory.
        """
        if "series_size_MB" not in series:
            return True
        return float(series["series_size_MB"].sum()) * 1e6 <= self.cap_bytes

    def make_room(self, incoming_bytes: int) -> int:
        """Evict least-recently-used series until `incoming_bytes` would fit.

        Returns how many bytes were freed. A single series larger than the whole
        cap is still fetched — refusing it would just deadlock the pipeline —
        but the cache is emptied first and the caller is told.
        """
        if incoming_bytes > self.cap_bytes:
            self._report("cache", f"series is {incoming_bytes / 1e9:.1f} GB, over the "
                                  f"{self.cap_bytes / 1e9:.0f} GB cap — emptying the cache for it")
            return self.evict_all()

        freed = 0
        while self.total_bytes + incoming_bytes > self.cap_bytes and self._entries:
            victim = min(self._entries.values(), key=lambda e: e.last_used)
            freed += self.evict(victim.series_uid)
        return freed

    def evict(self, series_uid: str) -> int:
        """Delete one cached series. Returns bytes freed."""
        entry = self._entries.pop(series_uid, None)
        directory = self.path_of(series_uid)
        size = entry.bytes if entry else _directory_size(directory)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        self._save_index()
        self._report("cache", f"evicted {series_uid[-12:]} ({size / 1e6:.0f} MB)")
        return size

    def evict_all(self) -> int:
        freed = sum(entry.bytes for entry in self._entries.values())
        for directory in self.root.iterdir():
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)
        self._entries.clear()
        self._save_index()
        return freed

    @property
    def total_bytes(self) -> int:
        return sum(entry.bytes for entry in self._entries.values())

    def status(self) -> dict:
        return {
            "root": self.root,
            "series": len(self._entries),
            "bytes": self.total_bytes,
            "cap_bytes": self.cap_bytes,
            "used_fraction": self.total_bytes / self.cap_bytes if self.cap_bytes else 0.0,
        }

    def rebuild_index(self) -> None:
        """Re-derive the index by scanning the cache directory.

        Used when the sidecar is missing or unreadable. Directories without any
        `.dcm` file are leftovers from an interrupted fetch and are removed.
        """
        self._entries = {}
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue  # the sidecar itself
            # `.partial` directories are interrupted fetches, and a directory
            # with no DICOM in it is the same thing by another name.
            files = [] if directory.name.endswith(".partial") else list(directory.rglob("*.dcm"))
            if not files:
                shutil.rmtree(directory, ignore_errors=True)
                continue
            stat = directory.stat()
            self._entries[directory.name] = CacheEntry(
                series_uid=directory.name,
                bytes=_directory_size(directory),
                instances=len(files),
                last_used=stat.st_mtime,
                fetched_at=stat.st_mtime,
            )
        self._save_index()

    # --------------------------------------------------------------- internal

    def _fetch(self, series_uid: str, expected_instances: int | None) -> Path:
        from idc_index import IDCClient  # slow import, and only needed on a miss

        final = self.path_of(series_uid)
        staging = self.root / f"{series_uid}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        self._report("download", f"fetching {series_uid[-12:]} from IDC")
        started = time.time()
        try:
            IDCClient().download_dicom_series(
                seriesInstanceUID=series_uid,
                downloadDir=str(staging),
                quiet=True,
                show_progress_bar=False,
                # Flatten: we key by series UID ourselves, so IDC's nested
                # collection/patient/study tree would only get in the way.
                dirTemplate="",
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        files = list(staging.rglob("*.dcm"))
        if not files:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"IDC returned no DICOM files for series {series_uid}")
        if expected_instances and len(files) < expected_instances:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(
                f"Incomplete download for {series_uid}: got {len(files)} of "
                f"{expected_instances} instances"
            )

        _replace_directory(staging, final)

        now = time.time()
        self._entries[series_uid] = CacheEntry(
            series_uid=series_uid,
            bytes=_directory_size(final),
            instances=len(files),
            last_used=now,
            fetched_at=now,
        )
        self._save_index()
        self._report("download", f"cached {series_uid[-12:]}: {len(files)} slices, "
                                 f"{self._entries[series_uid].bytes / 1e6:.0f} MB "
                                 f"in {now - started:.0f}s")
        return final

    def _touch(self, series_uid: str) -> None:
        self._entries[series_uid].last_used = time.time()
        self._save_index()

    def _index_path(self) -> Path:
        return self.root / INDEX_NAME

    def _load_index(self) -> dict[str, CacheEntry]:
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            raw = jsonio.read(path)
            entries = {e["series_uid"]: CacheEntry(**e) for e in raw.get("entries", [])}
        except Exception:
            # A corrupt sidecar is not worth a crash; the directories are truth.
            self.rebuild_index()
            return self._entries
        # Drop entries whose directory vanished underneath us.
        return {uid: e for uid, e in entries.items() if (self.root / uid).is_dir()}

    def _save_index(self) -> None:
        jsonio.write(
            self._index_path(),
            {"cap_bytes": self.cap_bytes, "entries": [e.as_dict() for e in self._entries.values()]},
            indent=None,
        )

    def _report(self, stage: str, message: str) -> None:
        if self.on_progress:
            self.on_progress(stage, message)


def _directory_size(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def _replace_directory(source: Path, target: Path, attempts: int = 8) -> None:
    """Move `source` onto `target`, retrying briefly on Windows sharing errors.

    Committing a download is a directory rename, and on Windows that is not
    reliably immediate. The `s5cmd` process that just wrote the files, and the
    indexer or scanner that noticed them appearing, can still hold handles for a
    moment after the download call returns — so the rename fails with
    `WinError 5: Access is denied` even though nothing is wrong. The same applies
    to deleting a directory: `rmtree` returns before the entries are necessarily
    gone, and renaming onto the path it just cleared can fail for the same reason.

    Retrying with a short backoff is the fix that matches the cause. Each attempt
    re-clears the target, because a partially-completed delete is exactly the
    state that makes the next rename fail.
    """
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            if target.exists():
                shutil.rmtree(target)
            source.rename(target)
            return
        except OSError as e:
            last_error = e
            time.sleep(0.25 * (attempt + 1))

    # Falling back to a copy loses the atomicity but keeps the download, which
    # took far longer to obtain than the copy will take.
    try:
        shutil.copytree(source, target, dirs_exist_ok=True)
        shutil.rmtree(source, ignore_errors=True)
        return
    except OSError:
        pass

    shutil.rmtree(source, ignore_errors=True)
    raise RuntimeError(
        f"Could not commit {source.name} to the cache after {attempts} attempts: {last_error}"
    ) from last_error


def primary_series(patient_id: str, series_table: Path | None = None) -> pd.DataFrame:
    """The one series per screening round that models should read, for a patient."""
    table = series_table or paths.SERIES_PARQUET
    if not Path(table).exists():
        raise SystemExit(f"No cohort at {table} — run `python -m cohort.build_cohort` first")
    df = pd.read_parquet(table)
    mine = df[(df["PatientID"].astype(str) == str(patient_id)) & df["is_primary"]]
    return mine.sort_values("round_key")


# ------------------------------------------------------------------------ CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", action="append", default=[], help="SeriesInstanceUID to fetch (repeatable)")
    p.add_argument("--patient", help="fetch every primary series for this patient")
    p.add_argument("--cap-gb", type=float, default=None, help="cache cap in GB (default: $CT_CACHE_GB or 50)")
    p.add_argument("--status", action="store_true", help="report what is cached and exit")
    p.add_argument("--evict-all", action="store_true", help="empty the cache and exit")
    p.add_argument("--rebuild-index", action="store_true", help="rescan the cache directory and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache = SeriesCache(
        cap_bytes=int(args.cap_gb * 1e9) if args.cap_gb else paths.cache_cap_bytes(),
        on_progress=lambda _stage, message: print(message, file=sys.stderr),
    )

    if args.rebuild_index:
        cache.rebuild_index()
    if args.evict_all:
        freed = cache.evict_all()
        print(f"Freed {freed / 1e9:.2f} GB")
        return 0

    if args.status or not (args.series or args.patient):
        status = cache.status()
        print(f"cache     {status['root']}")
        print(f"series    {status['series']:,}")
        print(f"size      {status['bytes'] / 1e9:.2f} GB of {status['cap_bytes'] / 1e9:.0f} GB "
              f"({status['used_fraction']:.0%})")
        return 0

    if args.patient:
        table = primary_series(args.patient)
        if table.empty:
            raise SystemExit(f"Patient {args.patient} has no primary series in the cohort")
        if not cache.fits_together(table):
            print(f"warning: this patient's {table['series_size_MB'].sum() / 1e3:.1f} GB exceeds the "
                  f"{cache.cap_bytes / 1e9:.0f} GB cap, so earlier rounds are evicted as later ones "
                  f"arrive", file=sys.stderr)
        # Printed as they arrive: a path is only guaranteed to exist until the
        # next series is fetched (see SeriesCache.stream).
        for directory in cache.stream(table):
            print(directory, flush=True)
    else:
        for uid in args.series:
            print(cache.get(uid), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
