"""Change detection has to find nodules and reject artefacts.

Registration itself is exercised against synthetic volumes rather than real CT:
the point is the filtering and the stage selection, which are where the false
positives live. §11 is explicit that registration artefacts masquerading as
tissue change is the failure mode that matters here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.change import (
    MIN_FILL_FRACTION,
    _side,
    alignment_error,
    find_regions,
    inflation_shift,
)

SPACING = 1.5


def empty(shape=(40, 60, 60)) -> np.ndarray:
    return np.full(shape, -850.0, dtype=np.float32)


def sphere(volume: np.ndarray, centre, radius_mm: float, value: float) -> np.ndarray:
    """Paint a solid sphere, the shape a nodule actually has."""
    z, y, x = np.ogrid[: volume.shape[0], : volume.shape[1], : volume.shape[2]]
    r = radius_mm / SPACING
    mask = ((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2) <= r ** 2
    volume[mask] = value
    return volume


def tube(volume: np.ndarray, value: float) -> np.ndarray:
    """A long thin structure — what a shifted vessel looks like in a difference."""
    volume[5:35, 30, 30] = value
    volume[5:35, 31, 30] = value
    return volume


class TestFindRegions:
    @staticmethod
    def run(baseline, followup, mask=None):
        delta = followup - baseline
        if mask is None:
            mask = np.ones_like(baseline, dtype=bool)
        return find_regions(delta, baseline, followup, mask, SPACING)

    def test_finds_a_new_solid_nodule(self):
        baseline = empty()
        followup = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-50.0)
        regions = self.run(baseline, followup)
        assert len(regions) == 1
        assert regions[0].kind == "new"
        assert regions[0].diameter_mm == pytest.approx(8.0, abs=2.0)

    def test_classifies_growth_when_tissue_was_already_there(self):
        baseline = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-700.0)
        followup = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-150.0)
        regions = self.run(baseline, followup)
        assert len(regions) == 1
        assert regions[0].kind == "growth"

    def test_ignores_a_region_that_lost_density(self):
        """Resolving inflammation is not a finding, and would bury real ones."""
        baseline = sphere(empty(), (20, 30, 30), radius_mm=5.0, value=-100.0)
        followup = empty()
        assert self.run(baseline, followup) == []

    def test_rejects_a_thin_elongated_structure(self):
        """A shifted vessel is a tube; a nodule is not."""
        baseline = empty()
        followup = tube(empty(), value=-50.0)
        assert self.run(baseline, followup) == []

    def test_rejects_a_sparse_branching_structure(self):
        """The vascular tree fills only a few percent of its bounding box."""
        baseline = empty()
        followup = empty()
        rng = np.random.default_rng(0)
        # Scatter isolated voxels across a large box: high extent, tiny fill.
        for _ in range(200):
            z, y, x = rng.integers([10, 20, 20], [30, 45, 45])
            followup[z, y, x] = -50.0
        regions = self.run(baseline, followup)
        assert all(r.fill_fraction >= MIN_FILL_FRACTION for r in regions)

    def test_rejects_a_region_too_small_to_act_on(self):
        baseline = empty()
        followup = sphere(empty(), (20, 30, 30), radius_mm=1.0, value=-50.0)
        assert self.run(baseline, followup) == []

    def test_rejects_a_lobe_sized_density_shift(self):
        """Atelectasis or a registration failure, not a screening nodule."""
        baseline = empty()
        followup = sphere(empty(), (20, 30, 30), radius_mm=25.0, value=-50.0)
        assert self.run(baseline, followup) == []

    def test_rejects_a_change_that_starts_as_solid_tissue(self):
        """A vessel getting denser is not new growth."""
        baseline = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-200.0)
        followup = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=200.0)
        assert self.run(baseline, followup) == []

    def test_rejects_a_change_that_stays_aerated(self):
        """A small density shift in still-aerated lung is noise at this scale."""
        baseline = empty()
        followup = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-620.0)
        assert self.run(baseline, followup) == []

    def test_respects_the_mask(self):
        baseline = empty()
        followup = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-50.0)
        mask = np.zeros_like(baseline, dtype=bool)
        mask[:10] = True  # the nodule is at z=20, outside the mask
        assert self.run(baseline, followup, mask) == []

    def test_regions_are_ranked_by_magnitude(self):
        baseline = empty(shape=(60, 60, 60))
        followup = empty(shape=(60, 60, 60))
        sphere(followup, (15, 30, 30), radius_mm=4.0, value=-100.0)
        sphere(followup, (45, 30, 30), radius_mm=7.0, value=-100.0)
        regions = self.run(baseline, followup)
        assert len(regions) == 2
        assert regions[0].diameter_mm > regions[1].diameter_mm

    def test_sensitivity_floor_is_about_five_millimetres(self):
        """Documents the measured floor, which is above MIN_DIAMETER_MM.

        Smoothing with a 1-voxel sigma erodes a blob only a few voxels across,
        so the nominal 4 mm limit is not what the detector actually achieves.
        Pinning it here means a change to the smoothing cannot quietly move the
        sensitivity without someone noticing.
        """
        baseline = empty()
        too_small = sphere(empty(), (20, 30, 30), radius_mm=2.0, value=-50.0)
        big_enough = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-50.0)
        assert self.run(baseline, too_small) == []
        assert len(self.run(baseline, big_enough)) == 1

    def test_reported_diameter_is_an_upper_estimate(self):
        """Smoothing spreads the edge outward, so sizes read 1-2 mm high."""
        baseline = empty()
        followup = sphere(empty(), (20, 30, 30), radius_mm=4.0, value=-50.0)
        reported = self.run(baseline, followup)[0].diameter_mm
        assert 8.0 <= reported <= 11.0  # true diameter is 8 mm

    def test_reports_the_most_affected_slice(self):
        baseline = empty()
        followup = sphere(empty(), (25, 30, 30), radius_mm=4.0, value=-50.0)
        assert self.run(baseline, followup)[0].slice_index == pytest.approx(25, abs=2)


class TestSide:
    def test_image_left_is_the_patients_right(self):
        """CT is viewed from the feet, so the sides are mirrored.

        Getting this backwards puts a finding in the wrong lung in the summary.
        """
        assert _side(10, 20, 100) == "right"
        assert _side(80, 90, 100) == "left"


class TestInflationShift:
    def test_measures_a_uniform_density_offset(self):
        delta = np.full((10, 10, 10), 30.0)
        mask = np.ones((10, 10, 10), dtype=bool)
        assert inflation_shift(delta, mask) == pytest.approx(30.0)

    def test_is_robust_to_focal_change(self):
        """The median must ignore a real nodule; a mean would absorb it."""
        delta = np.full((10, 10, 10), 20.0)
        delta[:2, :3, :3] = 800.0
        mask = np.ones((10, 10, 10), dtype=bool)
        assert inflation_shift(delta, mask) == pytest.approx(20.0)

    def test_an_empty_mask_is_zero_not_an_error(self):
        assert inflation_shift(np.ones((4, 4, 4)), np.zeros((4, 4, 4), dtype=bool)) == 0.0


class TestAlignmentError:
    def test_identical_volumes_align_perfectly(self):
        volume = empty()
        assert alignment_error(volume, volume, np.ones_like(volume, dtype=bool)) == 0.0

    def test_grows_with_misalignment(self):
        baseline = sphere(empty(), (20, 30, 30), radius_mm=6.0, value=0.0)
        near = sphere(empty(), (21, 30, 30), radius_mm=6.0, value=0.0)
        far = sphere(empty(), (30, 30, 30), radius_mm=6.0, value=0.0)
        mask = np.ones_like(baseline, dtype=bool)
        assert alignment_error(baseline, near, mask) < alignment_error(baseline, far, mask)

    def test_an_empty_mask_is_nan_not_zero(self):
        """Zero would read as a perfect alignment; NaN reads as no measurement."""
        volume = empty()
        assert np.isnan(alignment_error(volume, volume, np.zeros_like(volume, dtype=bool)))


class TestSummaryOfFailedRegistration:
    """When no stage aligns the rounds, the report must not list regions.

    A caveat printed under a list of regions is not enough — a reader looks at
    the list. Regions derived from unregistered volumes are breathing and
    position described in the vocabulary of growth, so there is no list at all.
    """

    @staticmethod
    def results(failed: bool, regions: list) -> dict:
        return {
            "risk": {"unavailable": True},
            "change": {
                "pairs": [{
                    "from": "T0", "to": "T1",
                    "regions": regions,
                    "n_regions": len(regions),
                    "registration": {
                        "registration_failed": failed,
                        "alignment_before_hu": 188.5,
                        "alignment_after_hu": 442.5 if failed else 86.8,
                        "deformable": not failed,
                    },
                }]
            },
        }

    @staticmethod
    def region() -> dict:
        return {
            "kind": "new", "side": "right", "delta_mm": 6.5, "slice": 122,
            "baseline_hu": -876.0, "followup_hu": 190.0,
        }

    def test_a_failed_pair_is_reported_as_unalignable(self):
        from ml.summary import compose

        text = compose(self.results(failed=True, regions=[]))
        assert "could not be aligned" in text
        assert "candidate region" not in text

    def test_a_successful_pair_reports_its_regions(self):
        from ml.summary import compose

        text = compose(self.results(failed=False, regions=[self.region()]))
        assert "1 candidate region" in text
        assert "6 mm across in the right lung" in text

    def test_the_disclaimer_is_always_present(self):
        from ml.summary import compose

        for failed in (True, False):
            assert "Not a medical device" in compose(self.results(failed, []))

    def test_no_risk_model_never_reads_as_a_low_score(self):
        from ml.summary import compose

        text = compose(self.results(failed=False, regions=[]))
        assert "No risk estimate was produced" in text
        assert "0.0%" not in text
