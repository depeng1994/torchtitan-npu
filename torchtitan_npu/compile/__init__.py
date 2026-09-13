# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-time extensions for NPU models."""

__all__ = ["PatternReplacement", "register_pre_aot_patterns", "setup_patterns"]

from .pattern_manager import setup_patterns
from .pattern_replacement import PatternReplacement, register_pre_aot_patterns
