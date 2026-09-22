"""Overlays must land on the right anatomy and rank the right slices.

An overlay that points a reader at the wrong slice is worse than no overlay, so
the per-slice ranking is tested directly rather than assumed from the pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.saliency import change_overlay, quantise, resample_overlay_to, top_slices


class TestQuantise:
    def test_maps_the_given_range_onto_the_byte_range(self):
        values = np.linspace(0, 100, 101).reshape(101, 1, 1)
        data, (lo, hi) = quantise(values, lo=0, hi=100)
        assert (lo, hi) == (0.0, 100.0)
        assert data.min() == 0 and data.max() == 255
        assert data.dtype == np.uint8

    def test_clips_outside_the_range(self):
        data, _ = quantise(np.array([[[-50.0, 150.0]]]), lo=0, hi=100)
        assert list(data.ravel()) == [0, 255]

    def test_a_flat_field_does_not_divide_by_zero(self):
        data, (lo, hi) = quantise(np.full((2, 2, 2), 7.0))
        assert data.max() == 0 and lo == hi == 7.0

    def test_nan_becomes_zero_rather_than_propagating(self):
        data, _ = quantise(np.array([[[np.nan, 1.0]]]), lo=0, hi=1)
        assert data.ravel()[0] == 0


class TestTopSlices:
    def test_picks_the_strongest_slices(self):
        profile = np.zeros(100)
        profile[[10, 40, 70]] = [3, 5, 4]
        assert top_slices(profile, k=3) == [40, 70, 10]

    def test_enforces_separation_so_the_list_is_not_one_structure(self):
        profile = np.zeros(100)
        profile[[50, 51, 52, 80]] = [9, 8, 7, 6]
        chosen = top_slices(profile, k=3, min_separation=5)
        assert chosen == [50, 80]

    def test_ignores_empty_slices(self):
        assert top_slices(np.zeros(50)) == []

    def test_returns_at_most_k(self):
        rng = np.random.default_rng(0)
        assert len(top_slices(rng.random(200), k=4, min_separation=1)) == 4


class TestChangeOverlay:
    def test_only_densification_is_shown(self):
        """A region that lost density must not light up (see ml/change.py)."""
        delta = np.zeros((6, 8, 8), dtype=np.float32)
        delta[2, 3, 3] = 600.0
        delta[4, 3, 3] = -600.0
        overlay = change_overlay(delta)
        # +600 HU on the fixed 100-800 scale lands at (600-100)/700 * 255 = 182.
        assert overlay.data[2, 3, 3] == pytest.approx(182, abs=1)
        assert overlay.data[4, 3, 3] == 0

    def test_uses_a_fixed_scale_so_patients_are_comparable(self):
        """Auto-scaling would make a quiet scan look as alarming as a loud one."""
        quiet = change_overlay(np.full((3, 4, 4), 120.0, dtype=np.float32))
        loud = change_overlay(np.full((3, 4, 4), 700.0, dtype=np.float32))
        assert quiet.value_range == loud.value_range == (100.0, 800.0)
        assert quiet.data.max() < loud.data.max()

    def test_a_mask_confines_the_overlay(self):
        delta = np.full((3, 4, 4), 500.0, dtype=np.float32)
        mask = np.zeros((3, 4, 4), dtype=bool)
        mask[1] = True
        overlay = change_overlay(delta, mask=mask)
        assert overlay.data[0].max() == 0
        assert overlay.data[1].max() > 0

    def test_ranks_slices_by_how_much_changed(self):
        """Extent matters, not just peak intensity: both patches saturate."""
        delta = np.zeros((30, 10, 10), dtype=np.float32)
        delta[20, :6, :6] = 500.0  # a big patch
        delta[5, :2, :2] = 500.0  # a small one
        assert change_overlay(delta).top_slices[:2] == [20, 5]


class TestResample:
    def test_shape_is_matched_exactly(self):
        overlay = change_overlay(np.zeros((10, 20, 20), dtype=np.float32))
        resized = resample_overlay_to(overlay, (25, 40, 40))
        assert tuple(resized.shape) == (25, 40, 40)
        assert resized.data.shape == (25, 40, 40)

    def test_an_identical_shape_is_returned_untouched(self):
        overlay = change_overlay(np.zeros((4, 4, 4), dtype=np.float32))
        assert resample_overlay_to(overlay, (4, 4, 4)) is overlay

    def test_ranking_survives_resampling(self):
        """The regression this guards against.

        Recomputing the profile from the resampled bytes ties almost every
        slice at 255 and degenerates the ranking into evenly-spaced indices
        counting down from the last slice.
        """
        delta = np.zeros((20, 16, 16), dtype=np.float32)
        delta[4, :10, :10] = 900.0  # saturates, and is the largest
        delta[15, :3, :3] = 900.0  # saturates too, but is much smaller
        overlay = change_overlay(delta)
        assert overlay.top_slices[0] == 4

        resized = resample_overlay_to(overlay, (60, 32, 32))
        # 4/20 of the way through 60 slices is about slice 12.
        assert resized.top_slices[0] == pytest.approx(12, abs=2)
        # And the list must not be a fixed-stride countdown from the end.
        strides = np.diff(resized.top_slices)
        assert not (len(strides) > 1 and len(set(strides.tolist())) == 1)

    def test_profile_length_matches_the_new_depth(self):
        overlay = change_overlay(np.zeros((7, 8, 8), dtype=np.float32))
        resized = resample_overlay_to(overlay, (13, 8, 8))
        assert len(resized.per_slice) == 13
