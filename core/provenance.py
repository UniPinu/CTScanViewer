"""Stamp every result with what produced it.

CONTEXT.md §11 is blunt about this: nothing here is a medical device, and every
results object has to say so in machine-readable form. `research_only` is not a
flag anyone may flip.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from functools import lru_cache

from core.paths import ROOT

DISCLAIMER = "Research prototype on NLST data. Not a medical device; not for clinical use."


@lru_cache(maxsize=1)
def code_rev() -> str:
    """Current git revision, or 'unknown' outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def stamp(model: str, **extra) -> dict:
    """The `provenance` block carried by results/<patient>.json."""
    return {
        "model": model,
        "code_rev": code_rev(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "research_only": True,
        "disclaimer": DISCLAIMER,
        **extra,
    }
