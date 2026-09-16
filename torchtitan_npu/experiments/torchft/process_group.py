# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HCCL process-group construction and lifecycle for the TorchFT experiment."""

from datetime import timedelta
from typing import Any, cast

import torch
from torch.distributed import ProcessGroup, Store
from torchft.futures import context_timeout
from torchft.process_group import ProcessGroupWrapper, _WorkAcceleratorTimeout
from torchft.utils import synchronize


class ProcessGroupHCCLEx(ProcessGroupWrapper):
    def __init__(self, timeout=timedelta(seconds=60)):
        super().__init__(timeout)
        self._errored: Exception | None = None

    def errored(self):
        synchronize()
        return self._errored

    def _run_context(self):
        return context_timeout(self.abort, self._timeout)

    def _wrap_work(self, work, opts):
        timeout = getattr(opts, "timeout", self._timeout)
        if timeout.total_seconds() <= 0:
            timeout = self._timeout
        return _WorkAcceleratorTimeout(self, work, timeout)

    def getBackendName(self) -> str:  # noqa: N802
        return "torchft-hccl"

    def _create_pg(self, store: Store, rank: int, world_size: int) -> ProcessGroup:
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL  # pyrefly: ignore [missing-import]

        self._errored = None
        options = ProcessGroupHCCL.Options()
        options._timeout = self._timeout
        options.group_id = f"torchft_quorum_{self._quorum_id}_rank_{self._group_rank}"
        if self._global_ranks:
            options.global_ranks_in_group = self._global_ranks
        backend = ProcessGroupHCCL(store, rank, world_size, options)
        backend._set_sequence_number_for_group()
        pg = ProcessGroup(store, rank, world_size)
        pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        pg._register_backend(torch.device("npu"), ProcessGroup.BackendType.CUSTOM, backend)
        return pg

    def abort(self, errored: bool = True) -> None:
        if errored:
            self._errored = RuntimeError("aborted")
        pg = self._pg
        if pg is not None:
            backend = cast("Any", pg._get_backend(torch.device("npu")))
            backend.abort()
            backend.shutdown()
            backend.clear_workmeta_list()
            self._pg = None

    def shutdown(self) -> None:
        self.abort(errored=False)
