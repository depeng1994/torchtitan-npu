# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 parallelization: EP/FSDP/FullAC assembly (CP1-only).

Derived from the DSV4 policy with the CP/MTP/compile branches removed:
V4.1 rejects CP>1, PP>1 and torch.compile in its config, so this
assembly wires only what the supported matrix needs.  The decoder FSDP
wrapper is the generic (non-MTP) helper.
"""

from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import _maybe_regional_inductor_backend
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh, validate_config


def _apply_compile_v41(
    model,
    *,
    compile_config: CompileConfig,
    parallel_dims: ParallelDims,
) -> None:
    """Apply torch.compile to each TransformerBlock with fullgraph=True.

    The vectorized DSA (sparse_attention._forward_packed) removed the
    data-dependent Python loops, so each block now captures as a single
    full graph; shapes stay static (dynamic=False).
    """
    import torch
    import torch._dynamo.config
    from torchtitan.tools.logging import logger

    torch._dynamo.config.capture_scalar_outputs = True
    # All layers share one forward code object and roles are resolved from
    # module structure, so layers of the same topology role share one graph
    # (~6 roles for the 40-layer topology instead of one graph per layer).
    # Keep the per-code-object recompile budget above the role count as
    # insurance (default 16 aborts fullgraph=True on recompile).
    n_layers = len(getattr(model, "layers", {})) or 32
    for cfg_name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, cfg_name):
            setattr(
                torch._dynamo.config,
                cfg_name,
                max(getattr(torch._dynamo.config, cfg_name), n_layers + 8),
            )
    # Written via setattr: torch declares this flag as a bare module-level
    # ``False`` (pyrefly infers Literal[False]), so a direct True assignment
    # fails the bad-assignment check.
    setattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True)  # noqa: B010

    # The whole-graph capture requires the vectorized DSA: the per-document
    # reference loop (the eager default, frozen loss trajectories) cannot be
    # traced under fullgraph=True.  It also requires the shared-inputs
    # dataflow (the eager default keeps the context side-channel).
    from . import attention as _attention
    from . import sparse_attention as _sparse_attention

    _sparse_attention._VECTORIZED = True
    _attention._SHARED_INPUTS = True

    backend = _maybe_regional_inductor_backend(model, compile_config.backend)

    for layer_id, transformer_block in model.layers.named_children():
        # dynamic=False: the vectorized DSA keeps every shape static; a
        # dynamic re-trace is neither expected nor supported here.
        transformer_block.compile(backend=backend, fullgraph=True, dynamic=False)

    logger.info("Compiling each V4.1 TransformerBlock with torch.compile (fullgraph=True)")


def apply_activation_checkpointing(model, ac_config, dump_folder):
    """Apply the selected policy plus any model-specific extension blocks."""
    policy = ac_config.build(dump_folder=dump_folder)
    policy.apply(model)
    model.apply_activation_checkpointing_extensions(policy)


def parallelize_deepseek_v41(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Parallelize V4.1: sharding config, AC, then the decoder FSDP wrap."""
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    elif parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if ac_config is not None:
        apply_activation_checkpointing(model, ac_config, dump_folder)

    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    if model_compile_enabled:
        _apply_compile_v41(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    model.apply_fsdp_extensions(
        dp_mesh=dp_mesh,
        training=training,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
    )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    return model
