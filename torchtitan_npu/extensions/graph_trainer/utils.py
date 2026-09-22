# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared graph and profiling utilities for NPU auto-overlap."""

from __future__ import annotations

import operator
import os
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.utils import _pytree
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_ASCEND_WORK_PATH = "ASCEND_WORK_PATH"
_NPU_AUTO_OVERLAP_DEBUG = "NPU_AUTO_OVERLAP_DEBUG"
_NODE_ID_META = "npu_auto_overlap_node_id"

_ALWAYS_METADATA_ONLY_TARGETS = {
    operator.getitem,
    torch.ops.aten.alias.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.expand.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.unbind.int,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
}


def is_npu_auto_overlap_debug_enabled() -> bool:
    return os.getenv(_NPU_AUTO_OVERLAP_DEBUG, "0") == "1"


def call_function_target(node: torch.fx.Node) -> Callable[..., Any]:
    """Return the callable target of an FX ``call_function`` node."""
    if node.op != "call_function" or not callable(node.target):
        raise AssertionError(f"expected an FX call_function with a callable target, got {node.op}: {node.target}")
    return node.target


def _custom_meta(node: torch.fx.Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return dict(custom) if isinstance(custom, dict) else {}


def assign_stable_node_tags(gm: torch.fx.GraphModule) -> None:
    """Use each node's pre-scheduling ordinal as its stable ID."""
    for ordinal, node in enumerate(gm.graph.nodes):
        custom = _custom_meta(node)
        custom[_NODE_ID_META] = ordinal
        node.meta["custom"] = custom


def node_id(node: torch.fx.Node) -> int | None:
    value = _custom_meta(node).get(_NODE_ID_META)
    return value if isinstance(value, int) else None


def canonical_order(gm: torch.fx.GraphModule) -> dict[torch.fx.Node, int]:
    result: dict[torch.fx.Node, int] = {}
    for node in gm.graph.nodes:
        stable_id = node_id(node)
        if stable_id is None:
            raise RuntimeError(f"FX node {node.name} has no auto-overlap node ID")
        result[node] = stable_id
    return result


def _metadata_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    return tuple(leaf for leaf in _pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor))


def _reshape_shares_input_storage(node: torch.fx.Node) -> bool:
    """Return whether FakeTensor metadata proves ``reshape`` is view-only."""
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return False
    inputs = _metadata_tensors(node.args[0].meta.get("val"))
    outputs = _metadata_tensors(node.meta.get("val"))
    if len(inputs) != 1:
        return False
    if len(outputs) == 1:
        try:
            if inputs[0].untyped_storage()._cdata == outputs[0].untyped_storage()._cdata:
                return True
        except (AttributeError, NotImplementedError, RuntimeError):
            pass
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    value = inputs[0]
    if value.layout != torch.strided:
        return False
    # Keep symbolic checks guard-free; unknown contiguity may require a copy.
    if any(statically_known_true(size == 0) for size in value.shape):
        return True
    expected_stride = 1
    for size, stride in reversed(tuple(zip(value.shape, value.stride(), strict=True))):
        if statically_known_true(size == 1):
            continue
        if not statically_known_true(stride == expected_stride):
            return False
        expected_stride *= size
    return True


def is_metadata_only_compute_node(node: torch.fx.Node) -> bool:
    """Return whether an FX call only changes tensor metadata."""
    if node.op != "call_function":
        return False
    if node.target in _ALWAYS_METADATA_ONLY_TARGETS:
        return True
    return node.target == torch.ops.aten.reshape.default and _reshape_shares_input_storage(node)


@contextmanager
def isolated_cann_profiler_work_path(prefix: str) -> Iterator[Path]:
    """Use an isolated ``ASCEND_WORK_PATH`` for internal CANN profiling.

    Collection and export must finish inside the context. The directory is
    removed afterwards unless ``NPU_AUTO_OVERLAP_DEBUG=1``; the original
    environment value is always restored.
    """

    previous = os.environ.get(_ASCEND_WORK_PATH)
    keep_profile = is_npu_auto_overlap_debug_enabled()
    with ExitStack() as stack:
        if keep_profile:
            rank = os.getenv("RANK", "unknown")
            temp_dir = tempfile.mkdtemp(prefix=f"{prefix}rank{rank}_pid{os.getpid()}_")
        else:
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=prefix))
        work_path = Path(temp_dir).resolve()
        os.environ[_ASCEND_WORK_PATH] = str(work_path)
        try:
            if keep_profile:
                logger.info("NPU auto-overlap benchmark profiling retained, path=%s", work_path)
            yield work_path
        finally:
            if previous is None:
                os.environ.pop(_ASCEND_WORK_PATH, None)
            else:
                os.environ[_ASCEND_WORK_PATH] = previous


def dump_fx_graph(
    gm: torch.fx.GraphModule,
    prefix: str,
) -> Path:
    """Save a readable FX graph as ``<prefix>_rank<rank>.py``."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    output_dir = Path.cwd() / "fx_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_rank{rank}.py"
    readable = gm.print_readable(
        print_output=False,
        include_stride=True,
        include_device=True,
        expanded_def=True,
    )
    output_path.write_text(readable, encoding="utf-8")
    logger.info("Saved FX graph to %s", output_path)
    return output_path
