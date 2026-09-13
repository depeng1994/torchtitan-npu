# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-time extensions for NPU models.

Public contract: ``PatternReplacement`` (pattern definition) and
``setup_patterns`` (the single policy entry point).  The additive
``register_pre_aot_patterns`` helper is intentionally NOT part of the public
API — pattern policy ownership belongs to ``pattern_manager`` alone.
"""

__all__ = ["PatternReplacement", "setup_patterns"]

from .pattern_manager import setup_patterns
from .pattern_replacement import PatternReplacement
