# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom-op boundary for ``cann_ops_transformer.ElasticBuffer`` Engram lookups.

The CANN Engram kernels are ordinary dispatcher ops with meta kernels, but the
Python API that drives them is a class: ``engram_fetch`` returns a callable that
yields the fetched rows plus an ``EngramFetchCtx`` dataclass for backward, and
``engram_fetch_grad`` narrows its output to the data-dependent number of unique
rows with a host sync. Neither can be traced.

This module puts both behind custom ops, the way ``deepep.py`` does for MoE
dispatch/combine: the buffer and the fetch context cross the boundary as opaque
references, the host sync stays inside an op body where it is invisible to the
tracer, and ``register_fake`` reports the shapes -- including the unique-row
count as a dynamic size.

The table itself is passed as ``weight`` purely to carry the autograd edge. The
kernel reads the shard that ``engram_write`` registered, not this tensor, so the
op never touches it.
"""

__all__ = [
    "EngramBufferHandle",
    "EngramFetchHandle",
    "EngramTableHandle",
    "engram_accumulate_sparse_grad",
    "engram_fetch",
    "engram_fetch_grad",
]

from typing import Any

import torch
from torch._library.opaque_object import (
    CustomClassBase,  # pyrefly: ignore [missing-module-attribute]
    register_opaque_type,
)


class EngramBufferHandle(CustomClassBase):
    """Opaque reference to the ``ElasticBuffer`` owned by an Engram table."""

    def __init__(self, value: Any = None):
        self.value: Any = value

    def __eq__(self, other):
        return isinstance(other, EngramBufferHandle) and self.value is other.value

    def __hash__(self):
        return 0 if self.value is None else id(self.value)

    def __fx_repr__(self):
        return "EngramBufferHandle()", {"EngramBufferHandle": EngramBufferHandle}


register_opaque_type(EngramBufferHandle, typ="reference")


class EngramFetchHandle(CustomClassBase):
    """Opaque wrapper carrying a CANN ``EngramFetchCtx`` to backward."""

    def __init__(self, value: Any = None):
        self.value: Any = value

    def __eq__(self, other):
        return isinstance(other, EngramFetchHandle) and self.value is other.value

    def __hash__(self):
        return 0 if self.value is None else id(self.value)

    def __fx_repr__(self):
        return "EngramFetchHandle()", {"EngramFetchHandle": EngramFetchHandle}


register_opaque_type(EngramFetchHandle, typ="reference")


@torch.library.custom_op("engram::fetch", mutates_args=())
def engram_fetch(
    weight: torch.Tensor,
    indices: torch.Tensor,
    buffer: EngramBufferHandle,
) -> tuple[torch.Tensor, EngramFetchHandle]:
    """Gather the requested rows from every EP owner through ElasticBuffer."""
    fetched, fetch_ctx = buffer.value.engram_fetch(indices.contiguous())()
    # The caller flattens token/head axes with ``view``, so return the same
    # contiguous layout F.embedding would.
    return fetched.contiguous(), EngramFetchHandle(value=fetch_ctx)


@engram_fetch.register_fake
def _engram_fetch_fake(
    weight: torch.Tensor,
    indices: torch.Tensor,
    buffer: EngramBufferHandle,
):
    # The rows land where the request came from, not where the table lives: a
    # host-offload shard is a CPU parameter, while the kernel writes device
    # memory. Taking the device from ``weight`` puts the whole Engram output on
    # CPU in the graph, which only surfaces later as a device mismatch in the
    # gate's linear.
    return (
        torch.empty(
            (indices.shape[0], weight.shape[1]),
            dtype=weight.dtype,
            device=indices.device,
        ),
        EngramFetchHandle(),
    )


@torch.library.custom_op("engram::fetch_grad", mutates_args=())
def engram_fetch_grad(
    grad_fetched: torch.Tensor,
    fetch: EngramFetchHandle,
    buffer: EngramBufferHandle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse the EP exchange, returning owner-local rows and their IDs."""
    return buffer.value.engram_fetch_grad(grad_fetched.contiguous(), fetch.value)


@engram_fetch_grad.register_fake
def _engram_fetch_grad_fake(
    grad_fetched: torch.Tensor,
    fetch: EngramFetchHandle,
    buffer: EngramBufferHandle,
):
    # EngramFetchGrad coalesces equal owner-local rows, so its row count is the
    # number of distinct rows this rank owns among the ones fetched -- known
    # only at runtime.
    num_unique = torch.library.get_ctx().new_dynamic_size()
    return (
        grad_fetched.new_empty((num_unique, grad_fetched.shape[1])),
        grad_fetched.new_empty((num_unique,), dtype=torch.int32),
    )


class EngramTableHandle(CustomClassBase):
    """Opaque reference to the host-offload table that owns the sparse gradient."""

    def __init__(self, value: Any = None):
        self.value: Any = value

    def __eq__(self, other):
        return isinstance(other, EngramTableHandle) and self.value is other.value

    def __hash__(self):
        return 0 if self.value is None else id(self.value)

    def __fx_repr__(self):
        return "EngramTableHandle()", {"EngramTableHandle": EngramTableHandle}


register_opaque_type(EngramTableHandle, typ="reference")


@torch.library.custom_op("engram::accumulate_sparse_grad", mutates_args="unknown")
def engram_accumulate_sparse_grad(
    grad_rows: torch.Tensor,
    local_ids: torch.Tensor,
    table: EngramTableHandle,
) -> torch.Tensor:
    """Hand one microbatch's owner-local rows to the Host-resident table.

    Everything data-dependent -- the device-to-host copy, the row-ID range
    check, building the sparse COO tensor -- happens here, where it runs
    eagerly on real shapes.

    The scalar it returns exists so the call has a consumer. The table is
    authoritative on CPU and its gradient never becomes a ``weight.grad``, so
    the accumulation is a side effect; a backward built only out of side
    effects produces nothing any consumer depends on, and a compiler is free to
    drop the whole subgraph. ``_EngramFetchSparseOffload`` returns this scalar
    as the gradient of a keepalive input, which keeps the chain live by data
    dependence rather than by an effect declaration.
    """
    table.value.accumulate_sparse_gradient(local_ids, grad_rows)
    return grad_rows.new_zeros(())


@engram_accumulate_sparse_grad.register_fake
def _engram_accumulate_sparse_grad_fake(
    grad_rows: torch.Tensor,
    local_ids: torch.Tensor,
    table: EngramTableHandle,
) -> torch.Tensor:
    return grad_rows.new_empty(())
