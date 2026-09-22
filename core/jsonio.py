"""JSON that survives numpy.

pandas and numpy hand us int64/float32/bool_ everywhere; the stdlib encoder
refuses all of them. One encoder, used by every stage that writes JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            # NaN/inf are not valid JSON; null is the honest rendering.
            return v if math.isfinite(v) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.as_posix()
        return super().default(o)


def dumps(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, cls=NumpyEncoder, indent=indent)


def write(path: Path, obj: Any, indent: int | None = 2) -> Path:
    """Write JSON atomically, so a crashed run never leaves a half-file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(obj, indent), encoding="utf-8")
    tmp.replace(path)
    return path


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
