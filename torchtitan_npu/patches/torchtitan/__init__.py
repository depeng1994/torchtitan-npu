# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .components import checkpoint  # noqa: F401, I001

# Apply the Trainer replacement before importing any graph-trainer module.
# GraphTrainer inherits Trainer at class-definition time, so importing it first
# would permanently bind its Config to the pre-EMA Trainer schema.
from . import trainer  # noqa: F401
from .components import metrics, optimizer, validate  # noqa: F401
from .distributed import context_parallel, full_dtensor, parallel_dims  # noqa: F401
from .distributed.flex_shard import dist_muon  # noqa: F401
from .experiments.graph_trainer import (  # noqa: F401
    chunked_loss,
    ep_chunk_concretization,
    ep_forward_accumulation,
    ep_overlap_shape_queries,
    ep_ready_nodes_dedup,
    ep_shape_live_out,
    graph_trainer_runtime_context,
)
from .models.common import decoder, moe, rope, token_dispatcher  # noqa: F401
