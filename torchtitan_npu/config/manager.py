# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register trainer config converters and install them before Tyro parsing."""

from functools import wraps

from torchtitan.config.manager import ConfigManager
from torchtitan.trainer import Trainer

from torchtitan_npu.config.converters import TrainerConfigConverter

# The EMATrainer monkey-patch (patches/torchtitan/trainer.py) replaces
# ``torchtitan.trainer.Trainer`` with ``EMATrainer`` at import time, so the
# config type seen by ``parse_args`` may be the original ``Trainer.Config``
# while the converter was registered with ``EMATrainer.Config``.  Capture both
# base types for exact-type matching — subclasses (e.g. a user's specialised
# trainer config) must never be rewritten.
_EMATRAINER_CONFIG: type | None = None
_ORIGINAL_TRAINER_CONFIG: type | None = None
try:
    _EMATRAINER_CONFIG = Trainer.Config  # EMATrainer.Config after the patch
    _ORIGINAL_TRAINER_CONFIG = Trainer.__mro__[1].Config  # original Trainer.Config
except (IndexError, AttributeError):
    _ORIGINAL_TRAINER_CONFIG = Trainer.Config  # fallback: same class

_original_load_config = ConfigManager._load_config

_CONFIG_CONVERTERS: dict[type[Trainer.Config], TrainerConfigConverter] = {}


def register_config_converter(
    config_type: type[Trainer.Config],
    converter: TrainerConfigConverter,
) -> None:
    """Register a converter for one exact Trainer config type."""
    _CONFIG_CONVERTERS[config_type] = converter


def _find_converter(config: Trainer.Config) -> TrainerConfigConverter | None:
    """Match ``config`` to a registered converter by exact type.

    Exact-type matching keeps user-defined Trainer.Config subclasses (e.g. a
    specialised trainer config) untouched: only the base ``Trainer.Config`` and
    the ``EMATrainer.Config`` base are candidates for conversion.
    """
    return _CONFIG_CONVERTERS.get(type(config))


@wraps(_original_load_config)
def _patched_load_config(self, args: list[str]) -> tuple[object, list[str]]:
    config, filtered_args = _original_load_config(self, args)
    # Exact-type check against both the patched and the original Trainer.Config,
    # since the EMATrainer monkey-patch creates a class-identity split.
    config_type = type(config)
    converter = _CONFIG_CONVERTERS.get(config_type)
    if converter is not None:
        config = converter.convert(config)
    return config, filtered_args


def apply() -> None:
    """Install the loader wrapper that dispatches registered config converters."""
    ConfigManager._load_config = _patched_load_config  # type: ignore[method-assign]


apply()