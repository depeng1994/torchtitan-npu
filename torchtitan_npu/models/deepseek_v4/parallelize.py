# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh, validate_config
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import (
    apply_compile as apply_graph_trainer_compile,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.deepseek_v3.mtp import apply_fsdp_to_mtp_decoder

from torchtitan_npu.models.deepseek_v4.model import GraphTrainerDeepSeekV4Model


def apply_activation_checkpointing(model, ac_config, dump_folder):
    """Apply the selected policy plus any model-specific extension blocks."""
    policy = ac_config.build(dump_folder=dump_folder)
    policy.apply(model)
    extension = getattr(model, "apply_activation_checkpointing_extensions", None)
    if extension is not None:
        extension(policy)


def parallelize_deepseek_v4(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Parallelize DSV4 without applying the generic 3-argument CP wrapper.

    DSV4 sparse attention has additional KV/indexer arguments; its own token
    dispatcher and AscendC path own CP handling.  Derived models may extend AC
    and FSDP through the version-neutral model hooks used below.
    """
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    elif parallel_dims.tp_enabled or parallel_dims.ep_enabled or parallel_dims.cp_enabled:
        model.parallelize(parallel_dims)

    if ac_config is not None:
        apply_activation_checkpointing(model, ac_config, dump_folder)

    if compile_config.enable and "model" in compile_config.components:
        from torchtitan.distributed.compile import apply_compile

        apply_compile(model, compile_config=compile_config, parallel_dims=parallel_dims)

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

    fsdp_extension = getattr(model, "apply_fsdp_extensions", None)
    if fsdp_extension is not None:
        fsdp_extension(
            dp_mesh=dp_mesh,
            training=training,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
        )

    apply_fsdp_to_mtp_decoder(
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


def annotate_deepseek_v4(model: GraphTrainerDeepSeekV4Model) -> None:
    """Attach EP-region and module-FQN annotations to the FX graph."""
    annotate_moe_ep_regions()
    annotate_module_fqns(model)


def parallelize_graph_trainer_deepseek_v4(
    model: GraphTrainerDeepSeekV4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    assert get_spmd_backend() != "spmd_types", "The GraphTrainer path does not yet support the spmd_types backend."

    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for DeepSeek V4 sparse attention in the GraphTrainer path."
        )

    annotate_deepseek_v4(model)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)  # pyrefly: ignore [bad-assignment]

    model = apply_graph_trainer_compile(  # pyrefly: ignore [bad-assignment]
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
