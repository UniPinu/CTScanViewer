"""Run Sybil on one CT series and print the result as JSON.

**This file runs in a different interpreter from the rest of the project.**

Sybil pins `python_requires = ">=3.8,<3.11"`, `torch==1.13.1`, `numpy==1.24.1`
and `pydicom==2.3.0`. The pipeline runs on Python 3.13 with numpy 2.x and
pydicom 3.x, so the two cannot share a process — and forcing them into one would
break both. Instead `ml/risk.py` invokes this script in Sybil's own environment
and reads JSON off stdout, the same subprocess-plus-NDJSON pattern the viewer
already uses to drive `ingest/fetch_patient.py`.

The consequence, and the rule for anyone editing this file: **import nothing
from the project.** Not `core`, not `preprocess`, nothing. Only the standard
library and what Sybil itself brings. Its sole contract with the rest of the
codebase is the JSON object it prints.

Set up the environment once:

    py -3.10 -m venv .venv-sybil
    .venv-sybil/Scripts/pip install sybil
    setx SYBIL_PYTHON "%CD%\\.venv-sybil\\Scripts\\python.exe"

Usage (normally called by ml/risk.py, not by hand):
    python -m ml.sybil_runner --dicom-dir data/cache/1.2.840... --out results/tmp
    python -m ml.sybil_runner --selftest      # report the environment, load nothing
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

#: Sybil's published ensemble. Individual members are sybil_1 … sybil_5.
DEFAULT_MODEL = "sybil_ensemble"

#: Sybil predicts cumulative risk at these horizons, in years.
HORIZONS = (1, 2, 3, 4, 5, 6)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Sybil on one CT series (runs inside the Sybil venv)")
    p.add_argument("--dicom-dir", type=Path, help="directory of .dcm files for ONE series")
    p.add_argument("--out", type=Path, help="directory for attention arrays and overlays")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Sybil model name (default: {DEFAULT_MODEL})")
    p.add_argument("--no-attentions", action="store_true", help="skip attention maps (faster)")
    p.add_argument("--selftest", action="store_true", help="report the environment and exit")
    return p.parse_args(argv)


def environment_report() -> dict:
    """What this interpreter is and whether Sybil can actually load here."""
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "sybil": None,
        "torch": None,
        "cuda": False,
        "ok": False,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
    except Exception as e:
        report["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        import sybil

        report["sybil"] = getattr(sybil, "__version__", "unknown")
        report["ok"] = True
    except Exception as e:
        report["sybil_error"] = f"{type(e).__name__}: {e}"
    return report


def dicom_paths(dicom_dir: Path) -> list[str]:
    """Every DICOM file in the directory.

    Sybil sorts the series itself from the DICOM headers, so the order handed
    over does not matter — but the *contents* must be exactly one series.
    """
    files = sorted(str(p) for p in dicom_dir.rglob("*.dcm"))
    if not files:
        raise SystemExit(f"No .dcm files under {dicom_dir}")
    return files


def run(args: argparse.Namespace) -> dict:
    from sybil import Serie, Sybil

    files = dicom_paths(args.dicom_dir)
    model = Sybil(args.model)
    serie = Serie(files)

    want_attentions = not args.no_attentions
    result = model.predict([serie], return_attentions=want_attentions)

    # scores[0] is this serie's cumulative risk at years 1..6.
    scores = [float(v) for v in result.scores[0]]
    out: dict = {
        "model": args.model,
        "n_slices": len(files),
        "scores": dict(zip((f"year_{h}" for h in HORIZONS), scores)),
        "scores_list": scores,
    }

    if want_attentions and args.out:
        out["attention"] = save_attentions(result, serie, args)
    return out


def save_attentions(result, serie, args: argparse.Namespace) -> dict:
    """Collate Sybil's attention into a voxel-aligned volume and save it.

    **The collation happens here, not on the other side of the seam.** Sybil's
    raw tensors are internal to its architecture and cannot be interpreted
    without knowing that architecture:

        image_attention_1   (5, 1, 25, 256)   5 ensemble members, 25 downsampled
                                              slices, and 256 = a *flattened*
                                              16x16 grid — not a volume
        volume_attention_1  (5, 1, 25)        one weight per downsampled slice

    Both are **logits**, so every value is negative. Turning them into something
    drawable means exponentiating, averaging over the ensemble, multiplying the
    per-pixel attention by its slice's weight, reshaping 256 back to 16x16, and
    trilinearly upsampling to the series' own geometry. Getting any of that
    wrong produces an overlay that looks plausible and points at the wrong
    tissue, which is worse than no overlay at all.

    `sybil.utils.visualization.collate_attentions` is that procedure, written by
    the model's authors, and it is what renders their reference overlays. Using
    it here means the seam carries an (n_slices, 512, 512) array aligned to the
    scan — something any consumer can quantise and draw without knowing anything
    about Sybil.
    """
    import numpy as np

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    attentions = result.attentions[0]

    saved: dict = {"dir": str(out_dir), "files": {}}

    # The raw tensors are kept too: they are small, and they are what anyone
    # debugging a suspicious overlay will want to look at.
    for key in ("image_attention_1", "volume_attention_1"):
        if key not in attentions:
            continue
        array = attentions[key]
        if hasattr(array, "detach"):  # a torch tensor
            array = array.detach().cpu().numpy()
        array = np.asarray(array, dtype="float32")
        path = out_dir / f"{key}.npy"
        np.save(path, array)
        saved["files"][key] = path.name
        saved[f"{key}_shape"] = list(array.shape)

    try:
        from sybil.utils.visualization import collate_attentions

        n_slices = len(serie.get_raw_images())
        collated = np.asarray(collate_attentions(attentions, n_slices), dtype="float32")
        # float16 halves a ~135 MB array at no visible cost: the source is a
        # 25x16x16 grid upsampled, so it carries nowhere near float32 detail.
        path = out_dir / "attention.npy"
        np.save(path, collated.astype("float16"))
        saved["files"]["attention"] = path.name
        saved["attention_shape"] = list(collated.shape)
        saved["attention_range"] = [float(collated.min()), float(collated.max())]
    except Exception as e:
        # Without this the overlay cannot be drawn, but the risk scores are
        # still valid — so report it rather than failing the whole prediction.
        saved["collate_error"] = f"{type(e).__name__}: {e}"

    # Sybil's own overlay renderer, kept because its heatmaps are the reference
    # rendering; ml/saliency.py produces the viewer's version alongside it.
    try:
        from sybil import visualize_attentions

        visualize_attentions(
            [serie], attentions=result.attentions,
            save_directory=str(out_dir / "overlays"), gain=3,
        )
        saved["overlays"] = "overlays"
    except Exception as e:  # rendering is a nicety; the arrays are the payload
        saved["overlay_error"] = f"{type(e).__name__}: {e}"

    return saved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.selftest:
        print(json.dumps(environment_report()))
        return 0

    if not args.dicom_dir:
        print(json.dumps({"error": "--dicom-dir is required"}))
        return 2

    try:
        print(json.dumps(run(args)))
        return 0
    except Exception as e:
        # The caller parses stdout, so a failure has to be JSON too. The
        # traceback goes to stderr, where risk.py logs it.
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
