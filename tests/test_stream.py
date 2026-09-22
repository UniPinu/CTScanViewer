"""The DICOM cache must stay inside its budget and never hand out a stale path.

These tests use a fake downloader rather than the network: what is being tested
is the eviction policy and the commit, not IDC.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ingest.stream import SeriesCache, _replace_directory


def fake_series(uid: str, mb: float, instances: int = 4) -> dict:
    return {"SeriesInstanceUID": uid, "series_size_MB": mb, "instanceCount": instances}


def populate(cache: SeriesCache, uid: str, mb: float, instances: int = 4) -> None:
    """Write a fake cached series of roughly `mb` megabytes."""
    directory = cache.path_of(uid)
    directory.mkdir(parents=True, exist_ok=True)
    payload = b"0" * int(mb * 1e6 / instances)
    for i in range(instances):
        (directory / f"{i:04d}.dcm").write_bytes(payload)
    cache.rebuild_index()


class TestBudget:
    def test_an_empty_cache_reports_zero(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        assert cache.total_bytes == 0
        assert cache.status()["series"] == 0

    def test_make_room_evicts_least_recently_used_first(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(25e6))
        for uid in ("a", "b", "c"):
            populate(cache, uid, mb=8)
        # Touch "a" so "b" becomes the least recently used.
        cache.get("a")

        cache.make_room(int(8e6))
        assert not cache.contains("b")
        assert cache.contains("a")
        assert cache.contains("c")

    def test_make_room_frees_enough_for_the_incoming_series(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(20e6))
        for uid in ("a", "b"):
            populate(cache, uid, mb=9)
        cache.make_room(int(15e6))
        assert cache.total_bytes + 15e6 <= cache.cap_bytes

    def test_a_series_larger_than_the_cap_empties_the_cache_rather_than_deadlocking(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(10e6))
        populate(cache, "a", mb=5)
        freed = cache.make_room(int(50e6))
        assert freed > 0
        assert cache.total_bytes == 0

    def test_fits_together_guards_the_change_head(self, tmp_path):
        """Registration needs two rounds resident at once."""
        cache = SeriesCache(root=tmp_path, cap_bytes=int(100e6))
        pair = pd.DataFrame([fake_series("a", 40), fake_series("b", 40)])
        assert cache.fits_together(pair)
        too_big = pd.DataFrame([fake_series("a", 80), fake_series("b", 80)])
        assert not cache.fits_together(too_big)


class TestIndex:
    def test_rebuild_finds_cached_series(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        populate(cache, "a", mb=2)
        assert cache.contains("a")
        assert cache.status()["series"] == 1

    def test_rebuild_discards_interrupted_downloads(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        partial = tmp_path / "abc.partial"
        partial.mkdir()
        (partial / "0000.dcm").write_bytes(b"0")
        cache.rebuild_index()
        assert not partial.exists()

    def test_rebuild_discards_directories_with_no_dicom(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        (tmp_path / "empty").mkdir()
        cache.rebuild_index()
        assert not (tmp_path / "empty").exists()

    def test_a_corrupt_sidecar_is_rebuilt_not_fatal(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        populate(cache, "a", mb=1)
        (tmp_path / "cache_index.json").write_text("{not json", encoding="utf-8")
        reopened = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        assert reopened.contains("a")

    def test_an_entry_whose_directory_vanished_is_dropped(self, tmp_path):
        import shutil

        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        populate(cache, "a", mb=1)
        shutil.rmtree(cache.path_of("a"))
        reopened = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        assert not reopened.contains("a")


class TestReplaceDirectory:
    """Committing a download is a directory rename, which Windows can refuse."""

    def test_moves_onto_a_fresh_target(self, tmp_path):
        source, target = tmp_path / "src", tmp_path / "dst"
        source.mkdir()
        (source / "a.dcm").write_bytes(b"x")
        _replace_directory(source, target)
        assert (target / "a.dcm").read_bytes() == b"x"
        assert not source.exists()

    def test_replaces_an_existing_target(self, tmp_path):
        source, target = tmp_path / "src", tmp_path / "dst"
        source.mkdir()
        (source / "new.dcm").write_bytes(b"new")
        target.mkdir()
        (target / "old.dcm").write_bytes(b"old")

        _replace_directory(source, target)
        assert (target / "new.dcm").exists()
        assert not (target / "old.dcm").exists()

    def test_retries_a_transient_sharing_error(self, tmp_path, monkeypatch):
        """The WinError 5 case: the first rename fails, a later one succeeds."""
        source, target = tmp_path / "src", tmp_path / "dst"
        source.mkdir()
        (source / "a.dcm").write_bytes(b"x")

        real_rename = type(source).rename
        calls = {"n": 0}

        def flaky(self, dest):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(5, "Access is denied")
            return real_rename(self, dest)

        monkeypatch.setattr(type(source), "rename", flaky)
        _replace_directory(source, target, attempts=5)
        assert (target / "a.dcm").exists()
        assert calls["n"] == 3

    def test_falls_back_to_a_copy_when_rename_never_succeeds(self, tmp_path, monkeypatch):
        source, target = tmp_path / "src", tmp_path / "dst"
        source.mkdir()
        (source / "a.dcm").write_bytes(b"x")

        def always_denied(self, dest):
            raise PermissionError(5, "Access is denied")

        monkeypatch.setattr(type(source), "rename", always_denied)
        # A download is expensive; keeping it via copy beats discarding it.
        _replace_directory(source, target, attempts=2)
        assert (target / "a.dcm").read_bytes() == b"x"


class TestStreaming:
    def test_stream_is_a_generator_not_a_list(self, tmp_path):
        """A list of paths would be a lie once the cache starts evicting."""
        import types

        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        result = cache.stream(pd.DataFrame([fake_series("a", 1)]))
        assert isinstance(result, types.GeneratorType)

    def test_stream_yields_cached_series_without_fetching(self, tmp_path):
        cache = SeriesCache(root=tmp_path, cap_bytes=int(1e9))
        populate(cache, "a", mb=1)
        populate(cache, "b", mb=1)
        yielded = list(cache.stream(pd.DataFrame([fake_series("a", 1), fake_series("b", 1)])))
        assert [p.name for p in yielded] == ["a", "b"]
        assert all(p.exists() for p in yielded)
