# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed NPU extensions to TorchTitan's training configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import tyro
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import CompileConfig as _BaseCompileConfig
from torchtitan.config import TrainingConfig as _BaseTrainingConfig
from torchtitan.tools.profiler import Profiler as _BaseProfiler

QuantizationRecipe = Literal["all_mxfp8", "mix", "all_block_fp8"]


@dataclass(frozen=True, slots=True)
class MuonOptimizerProfile:
    """Model-owned metadata required to construct DistMuon.

    The profile intentionally excludes scalar optimizer hyperparameters. Those
    are public CLI fields on :class:`OptimizerConfig` and are materialized only
    after Tyro has applied command-line overrides.
    """

    muon_pattern: str
    optimizer_factory_kwargs: Mapping[str, Mapping[str, Any]]


@dataclass(kw_only=True, slots=True)
class OptimizerConfig(OptimizersContainer.Config):
    """NPU optimizer CLI schema while preserving native optimizer configs."""

    name: Literal["native", "Muon"] = "native"
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_enable_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw", "spectral_unclamped"] = "match_rms_adamw"
    muon_ns_coefficients: tuple[float, float, float] = (
        3.4445,
        -4.7750,
        2.0315,
    )
    muon_eps: float = 1e-7
    _muon_profile: Annotated[MuonOptimizerProfile | None, tyro.conf.Suppress] = None

    def materialize(self) -> None:
        """Turn an explicit Muon selection into upstream optimizer groups.

        ``native`` is intentionally a strict no-op so converting every NPU
        recipe to this schema cannot alter its existing optimizer behavior.
        """
        if self.name == "native":
            return
        if self._muon_profile is None:
            raise ValueError("optimizer.name=Muon requires a recipe with a DSV4 Muon profile")

        self.param_groups = [
            ParamGroupConfig(
                pattern=self._muon_profile.muon_pattern,
                optimizer_name="DistMuon",
                optimizer_kwargs={
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "momentum": self.muon_momentum,
                    "nesterov": self.muon_enable_nesterov,
                    "ns_steps": self.muon_ns_steps,
                    "adjust_lr_fn": self.muon_adjust_lr_fn,
                    "ns_coefficients": self.muon_ns_coefficients,
                    "eps": self.muon_eps,
                    "fused": False,
                    "foreach": False,
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": self.lr,
                    "betas": (self.beta1, self.beta2),
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                    "fused": True,
                    "foreach": False,
                },
            ),
        ]
        self.optimizer_factory_kwargs_by_name = {
            name: dict(kwargs) for name, kwargs in self._muon_profile.optimizer_factory_kwargs.items()
        }


@dataclass(kw_only=True, slots=True)
class QuantizationExtensionConfig:
    """TorchAO-NPU quantized-training options.

    These fields define the public CLI schema. The quantization integration can
    consume them after CLI parsing without adding model-specific options to the
    upstream TorchTitan configuration.
    """

    enable_quantized_training: bool = False
    recipe: QuantizationRecipe = "mix"
    enable_mxfp4_qat: bool = False
    dst_type_max: float = 0.0


@dataclass(kw_only=True, slots=True)
class CompileExtensionConfig:
    """NPU-specific compile extension options.

    Controls automatic NPU pre-AOT pattern registration and the per-pattern
    blacklist.  ``enable_patterns`` and ``pattern_blacklist`` are CLI-settable
    via ``--compile.extension.enable-patterns`` etc.
    """

    enable_patterns: bool = True
    """Enable automatic NPU pre-AOT pattern registration."""

    pattern_blacklist: tuple[str, ...] = ()
    """Pattern names to skip even when available.  Use the stable pattern name
    (e.g. ``dsv4_partial_rope_wo_squeeze_forward``), not a Python module path."""


@dataclass(kw_only=True, slots=True)
class CompileConfig(_BaseCompileConfig):
    """NPU compile configuration with pattern-extension options.

    Extends upstream ``CompileConfig`` with ``extension`` so the CLI path is
    ``--compile.extension.enable-patterns`` etc.
    """

    extension: CompileExtensionConfig = field(
        default_factory=CompileExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class ExtensionConfig:
    """Global NPU extensions without an upstream component owner.

    Add a semantic group as a nested dataclass, then expose it with
    ``field(default_factory=...)``. For example::

        @dataclass(kw_only=True, slots=True)
        class RuntimeExtensionConfig:
            enable_feature: bool = False

        @dataclass(kw_only=True, slots=True)
        class ExtensionConfig:
            runtime: RuntimeExtensionConfig = field(
                default_factory=RuntimeExtensionConfig,
            )

    This produces the CLI option ``--extension.runtime.enable-feature``.
    """

    quantization: QuantizationExtensionConfig = field(
        default_factory=QuantizationExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class TrainingExtensionConfig:
    """NPU extensions owned by the training configuration."""

    allow_hf32: bool = True
    """Enable HF32 for the NPU matmul, convolution, and ACLNN backends."""


@dataclass(kw_only=True, slots=True)
class TrainingConfig(_BaseTrainingConfig):
    """Training options that are specific to NPU execution."""

    extension: TrainingExtensionConfig = field(
        default_factory=TrainingExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class ProfilerExtensionConfig:
    """NPU-specific options for the profiler component."""

    profiler_start: int | None = None
    """Absolute first training step to profile, inclusive."""

    profiler_end: int | None = None
    """Absolute training step at which profiling stops, exclusive."""

    profile_ranks: list[int] = field(default_factory=lambda: [-1])
    """Ranks to profile. ``[-1]`` profiles every rank."""

    profile_with_memory: bool = False
    """Whether to record memory events in the profiler trace."""

    profile_with_stack: bool = False
    """Whether to record Python/C++ stack information in the trace."""

    enable_online_parse: bool = True
    """Whether CANN should parse traces online via its trace handler."""


@dataclass(kw_only=True, slots=True)
class ProfilerConfig(_BaseProfiler.Config):
    """TorchTitan profiler configuration with NPU-specific extensions."""

    extension: ProfilerExtensionConfig = field(
        default_factory=ProfilerExtensionConfig,
    )
