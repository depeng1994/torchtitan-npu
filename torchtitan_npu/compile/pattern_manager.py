# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified discovery, filtering, and registration of NPU pre-AOT patterns.

Pattern modules export a module-level ``PATTERNS`` dict mapping pattern name to
``PatternReplacement``.  Import failure is tolerated so training can continue
without a particular pattern module.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from torchtitan_npu.compile.pattern_replacement import configure_pre_aot_patterns

if TYPE_CHECKING:
    from torchtitan_npu.compile.pattern_replacement import PatternReplacement

logger = logging.getLogger(__name__)

# Stable registration order: generic partial fusion patterns (including the
# attention-KV and compressor variants shared across the V4 model family)
# before the generic full-tensor fallback.
_BUILTIN_PATTERN_MODULES: tuple[str, ...] = (
    "torchtitan_npu.compile.patterns.common.partial_interleaved_rope",
    "torchtitan_npu.compile.patterns.common.interleaved_rope",
)


def _discover_builtin_patterns() -> dict[str, PatternReplacement]:
    """Import builtin pattern modules and collect their PATTERNS dicts.

    Pattern names are stable, user-facing identifiers: two modules exporting
    the same name is a registry invariant violation (a later module would
    silently shadow an earlier one and corrupt policy semantics), so it raises
    instead of silently overwriting.
    """
    patterns: dict[str, PatternReplacement] = {}
    for module_path in _BUILTIN_PATTERN_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.info("NPU pattern module %s skipped: %s", module_path, exc)
            continue
        module_patterns = getattr(module, "PATTERNS", None)
        if not module_patterns:
            continue
        duplicates = sorted(set(patterns) & set(module_patterns))
        if duplicates:
            raise ValueError(
                f"Duplicate NPU pattern name(s) across pattern modules: {duplicates} (module {module_path})"
            )
        patterns.update(module_patterns)
    return patterns


def setup_patterns(
    *,
    enable_patterns: bool = True,
    pattern_blacklist: tuple[str, ...] = (),
) -> None:
    """Discover, filter, and register NPU pre-AOT patterns.

    Call once after the final compile config is available and before the model
    is built / ``torch.compile`` is invoked.  Idempotent: re-invoking with the
    same policy does not duplicate the shared pass.
    """
    if not enable_patterns:
        configure_pre_aot_patterns({})
        logger.info("NPU compile patterns disabled; retaining decomposed Torch graph")
        return

    all_patterns = _discover_builtin_patterns()
    blacklist = frozenset(pattern_blacklist)

    skipped: list[str] = []
    if blacklist:
        skipped = [name for name in all_patterns if name in blacklist]
        selected = {name: p for name, p in all_patterns.items() if name not in blacklist}
        unknown = sorted(blacklist - set(all_patterns))
        for name in skipped:
            logger.info("NPU pattern %s skipped (blacklisted)", name)
        for name in unknown:
            logger.warning("NPU pattern %s blacklisted but not registered (typo?)", name)
    else:
        selected = all_patterns

    configure_pre_aot_patterns(selected)
    logger.info(
        "NPU compile patterns configured: %d active (blacklisted: %d)",
        len(selected),
        len(skipped),
    )
