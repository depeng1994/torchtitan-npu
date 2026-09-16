# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronous TorchFT manager for the NPU FSDP/EP training path."""

from dataclasses import dataclass
from datetime import timedelta

import torchft
from torchft.manager import Manager
from torchtitan.experiments.torchft.manager import TorchFTManager

from .process_group import ProcessGroupHCCLEx


class FTManagerEx(TorchFTManager):
    @dataclass(kw_only=True, slots=True)
    class Config(TorchFTManager.Config):
        enable: bool = True
        process_group: str = "hccl"

    def __init__(self, config: Config) -> None:
        if not config.enable or config.process_group != "hccl":
            raise ValueError("The NPU FT configuration requires enabled HCCL fault tolerance")
        if config.semi_sync_method is not None:
            raise ValueError("LocalSGD and DiLoCo are not supported by this synchronous configuration")
        self.group_size = config.group_size
        self.replica_id = config.replica_id
        # v0.3.0 uses this legacy flag to enable per-step hooks and loss sync.
        # It does not control the underlying Manager's quorum scheduling.
        self.use_async_quorum = True
        self.process_group = ProcessGroupHCCLEx(timedelta(milliseconds=config.process_group_timeout_ms))
        self._manager = Manager(
            pg=self.process_group,
            min_replica_size=config.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=False,
            replica_id=f"torchtitan_ft_{config.replica_id}",
            init_sync=True,
        )
        self.replicate_pg = torchft.process_group.ManagedProcessGroup(self._manager)
        self.replicate_pg.register("dp_replicate")
