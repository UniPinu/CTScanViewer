"""Stage 5 — assemble the plain-language summary (CONTEXT.md §4 Stage 5).

The summary is generated from a **deterministic template** filled from the
results object. That is a deliberate constraint, not a placeholder for something
cleverer: it is fully offline, reproducible, and structurally incapable of
inventing a number. §4 leaves the door open to passing the structured results to
an LLM for nicer prose later, with the same rule — the numbers come from the
template, never from the model.

Two rules shape the wording:

  * **Nothing is asserted that the pipeline did not compute.** If Sybil was not
    available, the summary says the risk model did not run. It never falls back
    to a hedge that reads like a low score.
  * **No diagnostic language.** §11 rules it out, and it would be wrong anyway:
    attention shows where a model looked, and a change region is a place to
    look, not a finding. The vocabulary here stays observational throughout.
"""

from __future__ import annotations

from core.provenance import DISCLAIMER

#: Rounds, spelled the way a reader thinks of them.
ROUND_NAMES = {"T0": "the baseline scan", "T1": "the first annual follow-up",
               "T2": "the second annual follow-up"}


def describe_round(label: str | None) -> str:
    return ROUND_NAMES.get(label or "", label or "an earlier round")


def risk_sentence(risk: dict | None) -> str:
    """The headline number, or an honest statement that there isn't one."""
    if not risk or risk.get("unavailable"):
        return (
            "No risk estimate was produced: the risk model is not installed in this "
            "deployment, so this report covers tissue change only."
        )

    year_1 = risk.get("year_1")
    if year_1 is None:
        return "No risk estimate was produced for this scan."

    text = f"Estimated 1-year lung-cancer risk is {year_1:.1%}"
    interval = risk.get("ci_1")
    if interval:
        text += f" (interval {interval[0]:.1%}–{interval[1]:.1%})"

    percentile = risk.get("cohort_percentile")
    if percentile is not None:
        comparison = "above" if percentile >= 50 else "below"
        text += f", {comparison} the median for this screening cohort ({percentile}th percentile)"

    if not risk.get("calibrated", False):
        text += ". The score is the model's own output, not yet recalibrated on this cohort"
    return text + "."


def horizon_sentence(risk: dict | None) -> str:
    """How the risk accumulates out to six years."""
    if not risk or risk.get("unavailable"):
        return ""
    curve = risk.get("curve")
    if not curve or len(curve) < 6:
        return ""
    return f"Cumulative risk rises to {curve[-1]:.1%} by year 6."


def attention_sentence(attention: dict | None) -> str:
    """Where the model looked — phrased as attention, never as a finding."""
    if not attention or not attention.get("top_slices"):
        return ""
    slices = attention["top_slices"][:3]
    listed = ", ".join(str(s) for s in slices)
    return (
        f"The model weighted slices {listed} most heavily; these show where it attended, "
        "which is not by itself evidence of disease."
    )


def change_sentence(change: dict | None) -> str:
    """What moved between rounds."""
    if not change or not change.get("pairs"):
        return ""

    parts = []
    for pair in change["pairs"]:
        regions = pair.get("regions") or []
        earlier = describe_round(pair.get("from"))
        later = describe_round(pair.get("to"))

        if (pair.get("registration") or {}).get("registration_failed"):
            parts.append(
                f"{later.capitalize()} could not be aligned with {earlier}, so no change "
                "regions are reported for that comparison."
            )
            continue

        if not regions:
            parts.append(
                f"Comparing {later} with {earlier}, no region met the change thresholds."
            )
            continue

        top = regions[0]
        verb = "appeared where none was visible before" if top.get("kind") == "new" else "enlarged"
        detail = (
            f"Comparing {later} with {earlier}, {len(regions)} candidate region"
            f"{'s' if len(regions) != 1 else ''} met the thresholds. The largest, "
            f"about {top['delta_mm']:.0f} mm across in the {top['side']} lung, {verb} "
            f"(density {top['baseline_hu']:.0f} to {top['followup_hu']:.0f} HU, "
            f"slice {top['slice']})."
        )
        parts.append(detail)
    return " ".join(parts)


def registration_caveat(change: dict | None) -> str:
    """Say so when the alignment the change map rests on was weak."""
    if not change or not change.get("pairs"):
        return ""
    for pair in change["pairs"]:
        report = pair.get("registration") or {}
        if report.get("registration_failed"):
            # Already stated by change_sentence; repeating it would be noise.
            continue
        if report.get("deformable") is False:
            return (
                "Deformable registration did not complete for this pair, so differences caused "
                "by breathing may not be fully removed."
            )
    return ""


def compose(results: dict) -> str:
    """Build the summary paragraph from a results object (§5.3).

    Every number in the output is read from `results`; none is computed here.
    That keeps this function a renderer, and keeps one place — the pipeline —
    responsible for arithmetic.
    """
    sentences = [
        risk_sentence(results.get("risk")),
        horizon_sentence(results.get("risk")),
        attention_sentence(results.get("attention")),
        change_sentence(results.get("change")),
        registration_caveat(results.get("change")),
    ]
    body = " ".join(s for s in sentences if s)
    return f"{body} {DISCLAIMER}".strip()
