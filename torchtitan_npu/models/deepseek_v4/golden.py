"""Which operator path the Golden contract selects.

``USE_GOLDEN`` is the single switch: it selects the reference operators (pure
Torch, the exact arithmetic the inference baseline uses) instead of the AscendC
kernels, and it also pins the dtype contract of the model output.

Read through :func:`golden_enabled` rather than caching it in a constant:
comparison scripts set the switch after their imports, so a value frozen at
import time would silently select the wrong path.  The older names
``TORCHTITAN_NPU_VISION_GOLDEN`` and ``TORCHTITAN_NPU_GOLDEN_TRAINING`` are
still honoured; both are subsumed by this one switch.
"""

from __future__ import annotations

import os


def golden_enabled() -> bool:
    """True when the reference operators are selected."""
    return os.getenv(
        "USE_GOLDEN",
        os.getenv("TORCHTITAN_NPU_VISION_GOLDEN", os.getenv("TORCHTITAN_NPU_GOLDEN_TRAINING", "0")),
    ) == "1"


__all__ = ["golden_enabled"]
