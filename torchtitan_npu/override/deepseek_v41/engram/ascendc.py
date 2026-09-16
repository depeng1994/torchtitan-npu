# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CANN Engram lookups with authoritative Host weights and sparse gradients.

The EP-local CPU table is excluded from FSDP. Its address-stable storage is
registered with ElasticBuffer, and only fetched rows enter NPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.profiler import record_function

from torchtitan_npu.models.deepseek_v41.engram_host import HostEngramTable
from torchtitan_npu.ops.ascendc.engram import EngramBufferHandle, EngramTableHandle

_ENGRAM_ALIGNMENT = 128
_INT32_MAX = torch.iinfo(torch.int32).max
_ENGRAM_GROUPS: dict[tuple[tuple[int, ...], int], dist.ProcessGroup] = {}


def _dedicated_engram_group(ep_group: dist.ProcessGroup, layer_id: int) -> dist.ProcessGroup:
    """Return a communicator used only by one table's ElasticBuffer.

    ``ElasticBuffer`` identifies its device-side context by the HCCL
    communicator name of the group it is built on, so two tables sharing one
    communicator collide: the second one fails with "Create HCCL context memory
    failed". Every Engram table therefore gets its own communicator, keyed by
    the EP rank set and the table's layer id.
    """
    ranks = tuple(dist.get_process_group_ranks(ep_group))
    key = (ranks, layer_id)
    group = _ENGRAM_GROUPS.get(key)
    if group is None:
        # EP groups are disjoint, and every member builds Engram tables in the
        # same model order. Local synchronization avoids requiring nonmembers
        # of this EP group to participate in communicator creation.
        new_group = dist.new_group(
            ranks=list(ranks),
            backend="hccl",
            use_local_synchronization=True,
        )
        assert isinstance(new_group, dist.ProcessGroup)
        group = new_group
        _ENGRAM_GROUPS[key] = group
    return group


class _EngramFetchSparseOffload(torch.autograd.Function):
    """Route owner-local sparse gradients to a Host-authoritative table.

    The CPU weight is an autograd input only to make the fetched NPU tensor
    differentiable. Backward does not expose ``weight.grad`` immediately:
    TorchTitan's EP gradient clipper currently requires every visible gradient
    to be a DTensor. Instead, the table accumulates the sparse CPU gradient and
    the Engram optimizer container attaches it immediately before SparseAdam.

    That accumulation is a side effect, which is why ``keepalive`` exists. A
    backward returning a gradient for none of its inputs computes nothing any
    consumer depends on, and a compiled backward drops it: under torch.compile
    the whole subgraph disappeared, the table never saw a gradient, and
    training silently continued with a frozen table. Returning the accumulation
    op's scalar as ``keepalive``'s gradient keeps the chain live by data
    dependence, which survives graph capture without an effect declaration.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        weight: torch.Tensor,
        row_ids: torch.Tensor,
        buffer: EngramBufferHandle,
        table: EngramTableHandle,
        keepalive: torch.Tensor,
    ) -> torch.Tensor:
        row_dim = weight.shape[1]
        with record_function("engram::fetch"):
            indices = row_ids.reshape(-1).to(dtype=torch.int32)
            fetched, fetch = torch.ops.engram.fetch(weight, indices, buffer)

        ctx.buffer = buffer
        ctx.fetch = fetch
        ctx.row_dim = row_dim
        ctx.table = table
        return fetched

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore [bad-override]
        grad_fetched = grad_output.reshape(-1, ctx.row_dim)
        with record_function("engram::fetch_grad"):
            grad_unique, unique_local_entry = torch.ops.engram.fetch_grad(
                grad_fetched,
                ctx.fetch,
                ctx.buffer,
            )
        with record_function("engram::sparse_accum"):
            # Normalization stays inside the op: the unique-row count is
            # data-dependent, so reshaping by it out here would be a guard.
            sink = torch.ops.engram.accumulate_sparse_grad(grad_unique, unique_local_entry, ctx.table)
        # The table's gradient stays off autograd because TorchTitan's EP
        # clipper requires every visible gradient to be a DTensor. Global
        # clipping still covers it: _clip_grad_norm_with_host_engram folds the
        # pending sparse gradient into the norm and scales it by the same
        # coefficient. ``sink`` is the keepalive gradient; see the class
        # docstring for why the backward must return one.
        return None, None, None, None, sink


class HostOffloadEngramTable(HostEngramTable):
    """Replace the Torch Host lookup with CANN Fetch/FetchGrad."""

    supports_model_compile = True

    @dataclass(kw_only=True, slots=True)
    class Config(HostEngramTable.Config):
        num_max_tokens_per_rank: int
        pin_memory: bool = True

    def __init__(self, config: Config):
        super().__init__(config)
        if config.num_max_tokens_per_rank <= 0:
            raise ValueError(f"num_max_tokens_per_rank must be positive, got {config.num_max_tokens_per_rank}.")
        if self.embedding_dim % _ENGRAM_ALIGNMENT != 0:
            # EngramFetch serves rows in 128-element units. Padding the shard to
            # meet that used to hide the requirement behind a per-forward copy;
            # declaring it keeps every flavor on the zero-copy path instead.
            raise ValueError(
                f"EngramFetch requires a row width that is a multiple of {_ENGRAM_ALIGNMENT}, "
                f"but the table declares embedding_dim={self.embedding_dim}."
            )
        if self.num_embeddings > _INT32_MAX:
            # EngramFetch takes int32 row IDs, so the table's row count is the
            # real constraint. Checking it here fails at build time instead of
            # waiting for a batch that happens to hash a row above the range.
            raise ValueError(
                f"EngramFetch only accepts int32 row IDs, but the table declares {self.num_embeddings} rows."
            )
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.pin_memory = config.pin_memory
        self._engram_storage_mapped = False
        self._elastic_buffer = None
        self._elastic_buffer_spec: tuple[int, int, torch.dtype, int] | None = None
        self._elastic_buffer_handle: EngramBufferHandle | None = None

        self._table_handle = EngramTableHandle(value=self)

    def _submit_engram_storage(self, elastic_buffer, weight: torch.Tensor) -> None:
        """Register the authoritative CPU shard, or refresh an older pack's mirror.

        Direct registration (ops-transformer PR 11226) retains this storage:
        SparseAdam updates it in place, so subsequent writes are unnecessary.
        The training loop completes a collective over the same ranks before
        the first Engram layer, ordering the previous optimizer step against
        the next lookup. Older packs still need a write on every forward.
        """
        storage = weight.detach()
        elastic_buffer.engram_write(storage)
        self._engram_storage_mapped = getattr(elastic_buffer, "_engram_storage_ref", None) is storage

    @staticmethod
    def _elastic_buffer_type():
        # Imported lazily so CPU/golden runs do not require the operator pack.
        from cann_ops_transformer import ElasticBuffer

        return ElasticBuffer

    def init_elastic_buffer(self, *, param_dtype: torch.dtype) -> None:
        """Create this table's communicator and ElasticBuffer.

        Called once from ``_shard_engram_tables``, which mirrors how
        ``BaseEPTokenDispatcher.wire_meshes`` calls ``init_buffer``: the EP mesh
        is known, no batch has been seen yet, and every rank reaches the
        collective in the same order. Doing it here rather than on the first
        lookup keeps ``dist.new_group`` out of forward, which a captured graph
        must not contain.

        Only the shard geometry is needed, all of which follows from the config,
        so this runs before ``to_empty`` materializes any storage.
        """
        if self.ep_mesh is None or self.ep_mesh.size() == 1:
            return
        ep_size = self.ep_mesh.size()
        if self.num_embeddings % ep_size != 0:
            raise ValueError(
                f"Engram table has {self.num_embeddings} physical rows, which is not divisible by EP degree {ep_size}."
            )
        local_rows = self.num_embeddings // ep_size
        dtype = self.storage_dtype(param_dtype=param_dtype)
        capacity = self.num_max_tokens_per_rank

        elastic_buffer_type = self._elastic_buffer_type()
        num_cpu_bytes = elastic_buffer_type.get_engram_storage_size_hint(local_rows, self.embedding_dim, dtype)
        group = _dedicated_engram_group(self.ep_mesh.get_group(), self.layer_id)
        self._elastic_buffer = elastic_buffer_type(
            group,
            num_cpu_bytes=num_cpu_bytes,
            num_max_tokens_per_rank=capacity,
            with_grad=True,
        )
        self._elastic_buffer_spec = (local_rows, self.embedding_dim, dtype, capacity)
        self._elastic_buffer_handle = EngramBufferHandle(value=self._elastic_buffer)

    def _assert_per_forward_write_allowed(self) -> None:
        """Fail with the cause when a per-forward host write meets compile.

        ``engram_write`` is an opaque host-side call, so a table that has to
        resubmit its shard on every forward cannot be captured. Raising while
        Dynamo traces turns an unexplained graph break into the actual reason.
        """
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "This Engram table resubmits its shard on every forward, which cannot be "
                "captured in a graph. Compiling it needs an operator package whose "
                "training-mode engram_write registers the storage directly "
                "(ops-transformer PR 11226), and a table whose storage is its own weight."
            )

    def _require_elastic_buffer(self, weight: torch.Tensor, request_count: int):
        """Return the buffer built for this shard, checking it still fits."""
        if self._elastic_buffer is None:
            raise RuntimeError(
                "The Engram ElasticBuffer has not been created. It is built by "
                "parallelize_deepseek_v41; a table used outside that path must call "
                "init_elastic_buffer() first."
            )
        capacity = self.num_max_tokens_per_rank
        if request_count > capacity:
            raise ValueError(
                "EngramFetch request count exceeds its fixed per-rank capacity: "
                f"got {request_count}, capacity {capacity}. Set "
                "num_max_tokens_per_rank to the largest "
                "batch * sequence * ngram-head count used by the run."
            )
        spec = (weight.shape[0], weight.shape[1], weight.dtype, capacity)
        if self._elastic_buffer_spec != spec:
            raise RuntimeError(
                "The Engram shard does not match the ElasticBuffer built for it: "
                f"{self._elastic_buffer_spec} -> {spec}."
            )
        return self._elastic_buffer

    def parallelize(self, parallel_dims) -> None:
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        if ep_mesh is None or ep_mesh.size() <= 1:
            raise ValueError("AscendC Host Engram requires expert_parallel_degree > 1.")
        super().parallelize(parallel_dims)

    def _init_self_parameters(self) -> None:
        super()._init_self_parameters()
        if self._elastic_buffer is not None:
            self._submit_engram_storage(self._elastic_buffer, self.weight)

    def _distributed_lookup(self, row_ids_N: torch.Tensor) -> torch.Tensor:
        assert self.ep_mesh is not None and self.ep_mesh.size() > 1
        flat_ids = row_ids_N.reshape(-1)
        elastic_buffer = self._require_elastic_buffer(self.weight, flat_ids.numel())
        if not self._engram_storage_mapped:
            self._assert_per_forward_write_allowed()
            with record_function("engram::host_write"):
                self._submit_engram_storage(elastic_buffer, self.weight)
        return _EngramFetchSparseOffload.apply(
            self.weight,
            flat_ids,
            self._elastic_buffer_handle,
            self._table_handle,
            self._grad_keepalive,
        ).view(*row_ids_N.shape, self.embedding_dim)


__all__ = ["HostOffloadEngramTable"]
