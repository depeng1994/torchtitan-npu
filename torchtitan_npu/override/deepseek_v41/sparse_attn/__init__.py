# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license in the repository.
"""V4.1 sparse-attention overrides; LI remains owned by its separate provider."""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v41.reference import ReferenceMetadataExtension
from torchtitan_npu.models.deepseek_v41.sparse_attention import V41SparseAttention


@override(target=ReferenceMetadataExtension.Config, exact=True, description="V4.1 SMLA metadata")
def asc_metadata(cfg: ReferenceMetadataExtension.Config):
    from .ascendc import AscV41MetadataExtension

    return derive(cfg, AscV41MetadataExtension.Config)


@override(target=V41SparseAttention.Config, exact=True, description="V4.1 SMLA forward and backward")
def asc(cfg: V41SparseAttention.Config):
    from .ascendc import AscV41SparseAttention

    return derive(cfg, AscV41SparseAttention.Config)
