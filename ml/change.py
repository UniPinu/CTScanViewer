"""Stage 3b — what changed in the lungs between two screening rounds (§2.4, §6.2).

This is the piece §2.4 calls the spine of the project. NLST scanned each
participant at baseline plus two annual follow-ups, so the question "did
anything grow?" is answerable directly from the data — and answerable *without
labels*, which is why it can ship while the CDAS application is still pending.

The hard part is not spotting differences. It is spotting the differences that
mean something. Between two scans a year apart, almost everything differs:

  * the patient is positioned differently in the scanner,
  * they inhaled a different amount, so the lungs are a different *shape*,
  * the scanner, kernel or dose may have changed,
  * and every voxel carries reconstruction noise of ±20–50 HU.

Subtracting two raw volumes therefore produces a picture of breathing, not of
disease. The pipeline here is built to strip each of those away in turn:

  1. **Resample** both rounds onto one isotropic grid, so a millimetre means the
     same thing in both and in every direction.
  2. **Mask to lung**, so the chest wall, the table and the arms cannot
     contribute — they move the most and matter the least.
  3. **Register** rigid -> affine -> deformable. Rigid absorbs positioning,
     affine absorbs overall scale, and the deformable (B-spline) stage absorbs
     breathing, which is the one that actually matters and the one a rigid-only
     pipeline gets wrong. **Each stage is then verified rather than trusted**:
     ITK returns wherever the optimiser stopped, which on real NLST pairs is
     sometimes worse than where it began. Every stage is resampled, scored by
     how well it actually aligns the lungs, and the best one is kept. Doing
     nothing competes too, so a pair that cannot be registered is reported as
     such instead of yielding a page of breathing artefacts.

     **This is not a theoretical safeguard.** On patient 110647 the stages
     scored `{bspline: 86.8, affine: 203.2, rigid: 347.7}` on one run against
     188.5 unregistered, and `{none: 188.5, rigid: 280.6, affine: 406.2,
     bspline: 406.4}` on the next — the same inputs, because ITK accumulates the
     metric across threads and the summation order varies. The optimiser is
     genuinely unstable on some pairs, and this pipeline's correctness cannot
     depend on it converging. See the README's known limitations.
  4. **Difference and filter.** Keep only regions that are large enough to see,
     dense enough to be tissue, and that grew *denser* — air becoming soft
     tissue is what a new or growing nodule looks like.

§11 is blunt that registration artefacts can masquerade as tissue change, so the
filters below are deliberately conservative and every region carries the numbers
that produced it. The output is candidate regions for a human to look at, not
findings.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import jsonio
from preprocess.volume import Series, lung_mask, resample_isotropic

#: Grid both rounds are resampled onto. 1.5 mm keeps nodules of interest
#: (>= 4 mm) several voxels wide while keeping registration to about a minute.
REGISTRATION_MM = 1.5

#: A region must gain at least this much density to count. Air-to-soft-tissue is
#: roughly +750 HU; 200 HU is well above reconstruction noise (±20–50 HU) while
#: still catching a part-solid nodule.
MIN_HU_DELTA = 200.0

#: Smallest region reported, as an equivalent-sphere diameter. Below ~4 mm,
#: screening guidelines do not act on a nodule and registration error dominates.
MIN_DIAMETER_MM = 4.0

#: Above this, the "region" is a lobe-sized density shift — atelectasis, an
#: infection, or a registration failure — not a screening nodule.
MAX_DIAMETER_MM = 30.0

#: The region must have been *air* before, at the 90th percentile of its voxels
#: rather than on average. A region straddling a vessel edge averages around
#: -600 HU and would pass a mean test while being entirely an artefact.
BASELINE_AIR_HU = -500.0

#: And it must be tissue now. Below this the "change" is a small density shift
#: in still-aerated lung, which at this scale is noise.
FOLLOWUP_TISSUE_HU = -600.0

#: Fraction of its bounding box a region must fill. This is the single most
#: effective artefact filter: the pulmonary vasculature, when it shifts by a
#: voxel, lights up as one enormous branching component filling ~3% of its box,
#: while a nodule is compact (a perfect sphere fills π/6 ≈ 0.52).
MIN_FILL_FRACTION = 0.30

#: Longest bounding-box side over shortest. Nodules are roughly isotropic;
#: vessel segments and pleural rims are tubes and sheets.
MAX_ELONGATION = 3.0


@dataclass
class ChangeRegion:
    """One candidate region of change, in the geometry of the earlier round."""

    slice_index: int  # most-affected slice, for the viewer to jump to
    bbox: list[int]  # [z0, y0, x0, z1, y1, x1] in voxels
    centroid_mm: list[float]
    volume_mm3: float
    diameter_mm: float
    mean_delta_hu: float
    peak_delta_hu: float
    baseline_hu: float  # mean density before; near air means "new"
    followup_hu: float
    fill_fraction: float  # how much of its bounding box the region fills
    elongation: float  # longest bounding-box side / shortest
    kind: str  # "new" | "growth"
    side: str  # "left" | "right" — the patient's own left/right

    def as_dict(self) -> dict:
        return {
            "slice": self.slice_index,
            "bbox": self.bbox,
            "centroid_mm": [round(v, 1) for v in self.centroid_mm],
            "volume_mm3": round(self.volume_mm3, 1),
            "delta_mm": round(self.diameter_mm, 1),
            "mean_delta_hu": round(self.mean_delta_hu, 1),
            "peak_delta_hu": round(self.peak_delta_hu, 1),
            "baseline_hu": round(self.baseline_hu, 1),
            "followup_hu": round(self.followup_hu, 1),
            "fill_fraction": round(self.fill_fraction, 2),
            "elongation": round(self.elongation, 1),
            "kind": self.kind,
            "side": self.side,
        }


@dataclass
class ChangeResult:
    """Everything one round-pair comparison produced."""

    from_round: str
    to_round: str
    regions: list[ChangeRegion] = field(default_factory=list)
    registration: dict = field(default_factory=dict)
    spacing_mm: float = REGISTRATION_MM
    shape: list[int] = field(default_factory=list)
    lung_volume_ml: tuple[float, float] = (0.0, 0.0)
    delta_path: Path | None = None
    interior_path: Path | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "from": self.from_round,
            "to": self.to_round,
            "regions": [r.as_dict() for r in self.regions],
            "n_regions": len(self.regions),
            "registration": self.registration,
            "spacing_mm": self.spacing_mm,
            "shape": self.shape,
            "lung_volume_ml": {
                "from": round(self.lung_volume_ml[0], 1),
                "to": round(self.lung_volume_ml[1], 1),
            },
            "seconds": round(self.seconds, 1),
        }


# ------------------------------------------------------------------ conversion


def to_sitk(volume: np.ndarray, spacing_mm: float):
    """Wrap a [z][y][x] array as a SimpleITK image with physical spacing.

    SimpleITK orders spacing (x, y, z) — the reverse of numpy's axis order.
    Getting this backwards silently scales the registration, so it is done in
    exactly one place.
    """
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    return image


def mask_to_sitk(mask: np.ndarray, spacing_mm: float):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetSpacing((spacing_mm, spacing_mm, spacing_mm))
    return image


# ---------------------------------------------------------------- registration


def register(fixed, moving, fixed_mask=None, moving_mask=None,
             deformable: bool = True, seed: int = 20260918) -> tuple[object, dict]:
    """Align `moving` onto `fixed`, coarse to fine. Returns (transform, report).

    Three stages, each initialised from the previous one:

      **Rigid (Euler3D)** — rotation and translation only. Undoes how the
      patient was lying. Six parameters, so it is fast and cannot distort
      anatomy even if it converges badly.

      **Affine** — adds scale and shear. Absorbs gross differences in lung
      inflation and any scanner scaling.

      **B-spline** — a smooth deformation field on a coarse control grid. This
      is what actually cancels breathing. The grid is kept coarse (about one
      control point per 5 cm) on purpose: a dense grid would be flexible enough
      to deform a growing nodule into its own baseline and erase the very signal
      we are looking for.

    Mattes mutual information is the metric throughout. The two rounds are the
    same modality but often different scanners, kernels and doses, and MI does
    not assume their intensities are related linearly.
    """
    import SimpleITK as sitk

    report: dict = {"stages": []}

    def new_method() -> "sitk.ImageRegistrationMethod":
        method = sitk.ImageRegistrationMethod()
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        # Sampling a fraction of voxels is much faster and, with a fixed seed,
        # still bit-for-bit reproducible.
        method.SetMetricSamplingStrategy(method.RANDOM)
        method.SetMetricSamplingPercentage(0.05, seed=seed)
        method.SetInterpolator(sitk.sitkLinear)
        if fixed_mask is not None:
            method.SetMetricFixedMask(fixed_mask)
        if moving_mask is not None:
            method.SetMetricMovingMask(moving_mask)
        return method

    def image_centre(image):
        """Physical coordinates of the middle of the volume.

        Every transform below is centred here. An affine or B-spline transform
        left at its default centre of (0, 0, 0) is centred on a *corner* of the
        volume, which gives its rotation and scale parameters lever arms tens of
        centimetres long; one optimiser step then swings the volume clear of the
        fixed image and the metric fails outright with "all samples map outside
        moving image buffer".
        """
        return image.TransformContinuousIndexToPhysicalPoint(
            [(size - 1) / 2.0 for size in image.GetSize()]
        )

    def run(method, stage_transform, prior, label: str):
        """Optimise one stage on top of everything already solved.

        The stages already solved are handed over with
        `SetMovingInitialTransform` rather than composed into the transform
        being optimised. ITK differentiates the transform it is optimising, and
        a CompositeTransform cannot supply a Jacobian with respect to position —
        composing first raises `ComputeJacobianWithRespectToPosition is
        unimplemented for CompositeTransform` the moment the optimiser starts.

        A stage that fails is recorded and skipped rather than aborting the
        comparison: a rigid-only alignment is still worth showing, as long as
        the report says that is what it is.
        """
        started = time.time()
        if prior is not None:
            method.SetMovingInitialTransform(prior)
        method.SetInitialTransform(stage_transform, inPlace=False)
        try:
            solved = method.Execute(fixed, moving)
        except RuntimeError as e:
            report["stages"].append({
                "stage": label,
                "stop": f"failed: {e}",
                "seconds": round(time.time() - started, 1),
            })
            report[f"{label}_error"] = str(e)
            return None
        report["stages"].append({
            "stage": label,
            "metric": round(float(method.GetMetricValue()), 5),
            "iterations": int(method.GetOptimizerIteration()),
            "stop": method.GetOptimizerStopConditionDescription(),
            "seconds": round(time.time() - started, 1),
        })
        return solved

    # Start from the centre of each volume, so the optimiser does not have to
    # travel the whole offset by gradient descent.
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    rigid_method = new_method()
    rigid_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    rigid_method.SetOptimizerScalesFromPhysicalShift()
    rigid_method.SetShrinkFactorsPerLevel([4, 2, 1])
    rigid_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    rigid_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    rigid = run(rigid_method, initial, None, "rigid")
    if rigid is None:
        # Nothing to build on. Return the centre-aligned initialisation, which
        # is still better than the raw geometry, and say the stage failed.
        report["final_metric"] = None
        report["deformable"] = False
        return [("initial", sitk.CompositeTransform([initial]))], report

    # Newest stage first: SimpleITK's CompositeTransform applies its entries
    # last-in-first-out, so index 0 is the transform applied last.
    solved = sitk.CompositeTransform([rigid])
    stages = [rigid]
    # Every stage is kept as a candidate, not just the last one. ITK returns
    # wherever the optimiser stopped, which is not necessarily better than where
    # it started — on real NLST pairs the B-spline stage sometimes terminates at
    # a *worse* metric than the affine result it was initialised from. The
    # caller resamples each candidate and keeps whichever actually aligns best.
    candidates: list[tuple[str, object]] = [("rigid", sitk.CompositeTransform([rigid]))]

    affine_method = new_method()
    # Plain gradient descent. A line search was tried here and made matters
    # worse on the hard pairs (the stage failed outright rather than merely
    # converging poorly), so the fixed step stays and the verification below is
    # what protects the result.
    affine_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=150, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    affine_method.SetOptimizerScalesFromPhysicalShift()
    affine_method.SetShrinkFactorsPerLevel([4, 2])
    affine_method.SetSmoothingSigmasPerLevel([2, 1])
    affine_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    affine_start = sitk.AffineTransform(3)
    affine_start.SetCenter(image_centre(fixed))
    affine = run(affine_method, affine_start, solved, "affine")
    if affine is not None:
        stages.insert(0, affine)
        solved = sitk.CompositeTransform(stages)
        candidates.append(("affine", solved))

    if deformable:
        # One control point per ~50 mm: enough to follow the diaphragm, too
        # coarse to deform a nodule away.
        size_mm = [sz * sp for sz, sp in zip(fixed.GetSize(), fixed.GetSpacing())]
        mesh = [max(3, int(round(extent / 50.0))) for extent in size_mm]
        bspline_method = new_method()
        bspline_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=60)
        # No optimiser scaling here: ITK's LBFGSB ignores scales and logs a
        # warning if any are set. That is part of why this stage is the one that
        # can terminate worse than it started, and why every stage is kept as a
        # candidate for the caller to verify.
        bspline_method.SetShrinkFactorsPerLevel([2, 1])
        bspline_method.SetSmoothingSigmasPerLevel([1, 0])
        bspline_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        try:
            bspline = run(
                bspline_method, sitk.BSplineTransformInitializer(fixed, mesh), solved, "bspline"
            )
            if bspline is not None:
                stages.insert(0, bspline)
                solved = sitk.CompositeTransform(stages)
                candidates.append(("bspline", solved))
                report["mesh"] = mesh
        except RuntimeError as e:
            # Deformable refinement is an improvement, not a requirement; the
            # affine result is still usable and honest about being affine-only.
            report["bspline_error"] = str(e)
            report["stages"].append({"stage": "bspline", "stop": f"failed: {e}"})

    report["completed_stages"] = [s["stage"] for s in report["stages"] if "metric" in s]
    report["final_metric"] = report["stages"][-1].get("metric")
    return candidates, report


def alignment_error(baseline: np.ndarray, other: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute HU difference inside the lung — how well two volumes line up.

    Reported before and after registration. It is the check that the transform
    was composed the right way round: a correct alignment drives this down, and
    an inverted or mis-composed one drives it *up*, which no metric value
    reported by the optimiser would reveal on its own.
    """
    if not mask.any():
        return float("nan")
    return float(np.abs(other[mask].astype(np.float32) - baseline[mask].astype(np.float32)).mean())


def resample_like(moving, fixed, transform, default_value: float = -1024.0):
    import SimpleITK as sitk

    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, default_value, moving.GetPixelID())


# ------------------------------------------------------------ change detection


def load_interior(out_dir: Path, shape: tuple[int, int, int]) -> np.ndarray | None:
    """Read back the eroded lung mask `compare` saved, unpacked to `shape`."""
    path = out_dir / "interior.npy"
    if not path.exists():
        return None
    packed = np.load(path)
    return np.unpackbits(packed, count=int(np.prod(shape))).astype(bool).reshape(shape)


def inflation_shift(delta: np.ndarray, mask: np.ndarray) -> float:
    """The uniform density offset caused by the two scans' different breath depth.

    Screening CT is taken at whatever inspiration the patient managed that day.
    Less air in the same tissue means every lung voxel reads denser, which is a
    real physical change and an uninteresting one. The median difference inside
    the lung estimates that offset robustly — the median ignores the heavy tail
    of genuine focal change, which a mean would absorb.
    """
    return float(np.median(delta[mask])) if mask.any() else 0.0


def find_regions(delta: np.ndarray, baseline: np.ndarray, followup: np.ndarray,
                 mask: np.ndarray, spacing_mm: float,
                 min_hu_delta: float = MIN_HU_DELTA,
                 min_diameter_mm: float = MIN_DIAMETER_MM,
                 max_diameter_mm: float = MAX_DIAMETER_MM) -> list[ChangeRegion]:
    """Turn a difference volume into a filtered list of candidate regions.

    Only *densification* is reported. A region losing density is usually
    resolving inflammation or a deeper breath, and surfacing it alongside growth
    would bury the signal a reader is looking for.

    Thresholding the difference alone is not close to sufficient. On a real
    NLST pair, +150 HU inside the lung mask selects ~8% of all lung voxels and
    yields hundreds of components, because a 1–2 mm residual shift at the edge
    of a pulmonary vessel swings a voxel from -850 HU to +50 HU. The shape
    filters below are what separate a nodule from that vasculature, and they do
    most of the work: on the pair this was developed against they take 645 raw
    components down to 11.

    Two measured consequences of the smoothing step, on synthetic spheres at
    1.5 mm spacing (see tests/test_change.py):

      * **The effective sensitivity floor is about 5 mm**, not the 4 mm
        `MIN_DIAMETER_MM` nominally allows. A Gaussian with a 1-voxel sigma
        erodes a blob only two or three voxels across, so the thresholded
        remnant falls below the size or compactness limits.
      * **Reported diameters run 1–2 mm high** for the same reason in reverse:
        smoothing spreads the region's edge outward, so a 7 mm sphere is
        reported at about 8 mm. `delta_mm` is therefore an upper estimate, and
        should not be compared directly against a radiologist's caliper
        measurement.

    The thresholds are physically motivated but **not yet validated against
    annotated positives** — §6.2 calls for exactly that, using the NLSTseg masks
    and Sybil boxes, and until it is done these are sensible defaults rather
    than tuned ones.
    """
    from scipy import ndimage

    voxel_mm3 = spacing_mm ** 3
    min_volume = (4 / 3) * np.pi * (min_diameter_mm / 2) ** 3
    max_volume = (4 / 3) * np.pi * (max_diameter_mm / 2) ** 3

    # Smooth before thresholding: isolated noisy voxels are not regions, and a
    # 1-voxel sigma is well below the size of anything reportable.
    smoothed = ndimage.gaussian_filter(delta.astype(np.float32), sigma=1.0)
    candidate = (smoothed > min_hu_delta) & mask
    if not candidate.any():
        return []

    labels, count = ndimage.label(candidate)
    regions: list[ChangeRegion] = []

    for index, bounds in enumerate(ndimage.find_objects(labels), start=1):
        blob = labels[bounds] == index
        volume_mm3 = float(blob.sum()) * voxel_mm3
        if not (min_volume <= volume_mm3 <= max_volume):
            continue

        # Shape first: it is the cheapest test and rejects the most.
        fill_fraction = float(blob.sum()) / float(blob.size)
        if fill_fraction < MIN_FILL_FRACTION:
            continue
        extents = sorted((s.stop - s.start) * spacing_mm for s in bounds)
        elongation = extents[2] / max(extents[0], 1e-6)
        if elongation > MAX_ELONGATION:
            continue

        baseline_voxels = baseline[bounds][blob]
        # 90th percentile, not the mean: a region straddling a vessel edge has a
        # mean near -600 HU but a 90th percentile up in soft tissue.
        if float(np.percentile(baseline_voxels, 90)) >= BASELINE_AIR_HU:
            continue

        after = float(followup[bounds][blob].mean())
        if after <= FOLLOWUP_TISSUE_HU:
            continue

        before = float(baseline_voxels.mean())
        deltas = smoothed[bounds][blob]
        diameter = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)

        z0, y0, x0 = (s.start for s in bounds)
        z1, y1, x1 = (s.stop for s in bounds)
        # The slice a reader should be shown is the one with the most change,
        # not the geometric middle of the blob.
        per_slice = (blob * np.abs(smoothed[bounds])).sum(axis=(1, 2))
        peak_slice = z0 + int(np.argmax(per_slice))

        centroid = ndimage.center_of_mass(blob)
        centroid_mm = [float((c + start) * spacing_mm) for c, start in zip(centroid, (z0, y0, x0))]

        regions.append(ChangeRegion(
            slice_index=peak_slice,
            bbox=[int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)],
            centroid_mm=centroid_mm,
            volume_mm3=volume_mm3,
            diameter_mm=diameter,
            mean_delta_hu=float(deltas.mean()),
            peak_delta_hu=float(deltas.max()),
            baseline_hu=before,
            followup_hu=after,
            fill_fraction=fill_fraction,
            elongation=elongation,
            # Below -800 HU the site was essentially air, so whatever is there
            # now is new rather than enlarged.
            kind="new" if before < -800 else "growth",
            side=_side(x0, x1, delta.shape[2]),
        ))

    # Biggest density change first: that is the order a reader wants to work in.
    regions.sort(key=lambda r: r.mean_delta_hu * r.volume_mm3, reverse=True)
    return regions


def _side(x0: int, x1: int, width: int) -> str:
    """Patient's own left/right.

    CT is displayed as if looking at the patient from their feet, so the
    patient's right lung appears on the left of the image. Getting this backwards
    would put a finding in the wrong lung in the summary text.
    """
    return "right" if (x0 + x1) / 2 < width / 2 else "left"


# -------------------------------------------------------------------- pipeline


def compare(earlier: Series, later: Series, from_round: str = "T0", to_round: str = "T1",
            spacing_mm: float = REGISTRATION_MM, deformable: bool = True,
            out_dir: Path | None = None) -> ChangeResult:
    """Full comparison of two rounds: resample, mask, register, difference, filter."""
    started = time.time()

    earlier_iso = resample_isotropic(earlier, spacing_mm)
    later_iso = resample_isotropic(later, spacing_mm)

    fixed_mask_array = lung_mask(earlier_iso.hu)
    moving_mask_array = lung_mask(later_iso.hu)

    fixed = to_sitk(earlier_iso.hu, spacing_mm)
    moving = to_sitk(later_iso.hu, spacing_mm)
    candidates, report = register(
        fixed, moving,
        mask_to_sitk(fixed_mask_array, spacing_mm),
        mask_to_sitk(moving_mask_array, spacing_mm),
        deformable=deformable,
    )

    import SimpleITK as sitk

    baseline = earlier_iso.hu.astype(np.float32)

    # Choose the stage that actually aligned best, rather than assuming the last
    # one did. The optimiser's own metric is not a safe guide here: it is
    # computed on a sampled subset under a mask, and a stage can terminate at a
    # worse value than it started from. Mean absolute HU difference inside the
    # lung is the quantity the change map is built on, so that is what is
    # compared, on the full volume.
    identity = sitk.Transform(3, sitk.sitkIdentity)
    unregistered = sitk.GetArrayFromImage(
        sitk.Resample(moving, fixed, identity, sitk.sitkLinear, -1024.0)
    )
    before = alignment_error(baseline, unregistered, fixed_mask_array)

    # "none" competes alongside the real stages. Registration is supposed to
    # help, and when it does not — which happens on genuinely hard pairs — using
    # a transform that demonstrably worsens alignment is strictly worse than
    # using none at all. Including it makes `alignment_after <= alignment_before`
    # a guarantee rather than a hope.
    scored = [(before, "none", unregistered)]
    for label, candidate in candidates:
        resampled = sitk.GetArrayFromImage(resample_like(moving, fixed, candidate))
        scored.append((alignment_error(baseline, resampled, fixed_mask_array), label, resampled))
    scored.sort(key=lambda item: item[0])
    best_error, best_label, warped = scored[0]

    delta = warped - baseline

    report["alignment_by_stage_hu"] = {label: round(err, 1) for err, label, _ in scored}
    report["selected_stage"] = best_label
    report["alignment_before_hu"] = round(before, 1)
    report["alignment_after_hu"] = round(best_error, 1)
    report["deformable"] = best_label == "bspline"
    report["registration_failed"] = best_label == "none"
    report["alignment_improved"] = report["alignment_after_hu"] < report["alignment_before_hu"]

    # Erode the mask a little: the lung boundary is where registration error is
    # largest, and a rim of false "growth" around the pleura would swamp
    # everything real inside it.
    from scipy import ndimage

    interior = ndimage.binary_erosion(fixed_mask_array, structure=np.ones((3, 3, 3)), iterations=2)

    # Remove the uniform density offset from differing breath depth before
    # hunting for focal change; otherwise a shallower second breath makes the
    # whole lung look like it gained tissue.
    shift = inflation_shift(delta, interior)
    report["inflation_shift_hu"] = round(shift, 1)

    if report["registration_failed"]:
        # Every candidate transform made alignment worse, so this difference map
        # is of two volumes that were never brought together. Anything found in
        # it would be breathing and position, presented in the vocabulary of
        # growth. A caveat under a list of regions is not enough — a reader
        # looks at the list. So there is no list.
        regions: list[ChangeRegion] = []
        report["note"] = (
            "Registration failed: no stage aligned these rounds better than leaving them "
            "unregistered, so no change regions are reported for this pair."
        )
    else:
        regions = find_regions(delta - shift, baseline, warped, interior, spacing_mm)
    voxel_ml = (spacing_mm ** 3) / 1000.0

    result = ChangeResult(
        from_round=from_round,
        to_round=to_round,
        regions=regions,
        registration=report,
        spacing_mm=spacing_mm,
        shape=[int(n) for n in baseline.shape],
        lung_volume_ml=(float(fixed_mask_array.sum()) * voxel_ml,
                        float(moving_mask_array.sum()) * voxel_ml),
        seconds=time.time() - started,
    )

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        # int16 keeps the difference map small enough to ship to the browser
        # while preserving 1 HU resolution.
        np.save(out_dir / "delta.npy", np.clip(delta, -1024, 1024).astype(np.int16))
        np.save(out_dir / "registered.npy", warped.astype(np.int16))
        # The mask the regions were found inside. The overlay must be confined to
        # the same voxels, or the picture shows densification along the skin,
        # chest wall and scanner table — where registration error is largest and
        # where no region was reported — and contradicts the findings beside it.
        np.save(out_dir / "interior.npy", np.packbits(interior, axis=None))
        result.interior_path = out_dir / "interior.npy"
        result.delta_path = out_dir / "delta.npy"
        jsonio.write(out_dir / "change.json", result.as_dict())

    return result


# ------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--earlier", type=Path, required=True, help="DICOM directory for the earlier round")
    p.add_argument("--later", type=Path, required=True, help="DICOM directory for the later round")
    p.add_argument("--out", type=Path, default=None, help="write delta.npy and change.json here")
    p.add_argument("--spacing", type=float, default=REGISTRATION_MM, help=f"grid in mm (default: {REGISTRATION_MM})")
    p.add_argument("--rigid-only", action="store_true", help="skip the deformable stage (faster, worse)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from preprocess.volume import load_series

    args = parse_args(argv)
    earlier = load_series(args.earlier)
    later = load_series(args.later)

    result = compare(
        earlier, later,
        from_round=earlier.meta.get("round") or "earlier",
        to_round=later.meta.get("round") or "later",
        spacing_mm=args.spacing,
        deformable=not args.rigid_only,
        out_dir=args.out,
    )

    print(f"{result.from_round} -> {result.to_round}  ({result.seconds:.0f}s)")
    for stage in result.registration["stages"]:
        print(f"  {stage['stage']:<8} metric {stage.get('metric')}  {stage.get('seconds', 0)}s")
    report = result.registration
    print(f"  by stage  {report.get('alignment_by_stage_hu')}  -> kept {report.get('selected_stage')}")
    print(f"  alignment {report['alignment_before_hu']} -> {report['alignment_after_hu']} HU "
          f"({'improved' if report['alignment_improved'] else 'WORSE — registration failed'})")
    print(f"  lung {result.lung_volume_ml[0]:.0f} -> {result.lung_volume_ml[1]:.0f} mL")
    print(f"\n{len(result.regions)} candidate region(s):")
    for i, region in enumerate(result.regions[:10], 1):
        print(f"  {i}. {region.kind:<7} {region.side:<5} lung  {region.diameter_mm:4.1f} mm  "
              f"slice {region.slice_index:>3}  {region.baseline_hu:>7.0f} -> {region.followup_hu:>7.0f} HU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
