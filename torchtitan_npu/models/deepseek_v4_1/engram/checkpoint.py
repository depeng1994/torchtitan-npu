# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU Engram shards in HF checkpoints, without gathering the Host table."""

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import _getdtype
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from torch.distributed.checkpoint.quantized_hf_storage import QuantizedHuggingFaceStorageReader


class EngramCheckpointTensor(torch.Tensor):
    """Expose a CPU row shard to DCP using global, unpadded HF coordinates.

    Unlike a DTensor this storage has no device mesh or tensor collectives.
    DCP's checkpointable tensor protocol handles the file offsets directly.
    Keep the padded backing tensor so a load can return the native EP shard.
    """

    @staticmethod
    def __new__(cls, weight: torch.Tensor, *, key: str, offset: int, logical_rows: int):
        result = torch.Tensor._make_wrapper_subclass(
            cls, (logical_rows, weight.shape[1]), dtype=weight.dtype, device=weight.device, requires_grad=False
        )
        return result

    def __init__(self, weight: torch.Tensor, *, key: str, offset: int, logical_rows: int):
        self.weight = weight
        self.key = key
        self.offset = min(offset, logical_rows)
        self.logical_rows = logical_rows

    @property
    def rows(self):
        return self.weight[: max(0, min(self.weight.shape[0], self.logical_rows - self.offset))]

    @classmethod
    def __torch_dispatch__(  # pyrefly: ignore [bad-param-name-override]
        cls, func, types, args: tuple[Any, ...] = (), kwargs=None
    ):
        kwargs = kwargs or {}
        if func not in (torch.ops.aten.detach.default, torch.ops.aten.clone.default, torch.ops.aten._to_copy.default):
            raise NotImplementedError(f"{func} is not a checkpoint storage operation")
        source = args[0]
        weight = func(source.weight, *args[1:], **kwargs)
        return cls(weight, key=source.key, offset=source.offset, logical_rows=source.logical_rows)

    def __create_chunk_list__(self):
        if self.rows.shape[0] == 0:
            return []
        return [ChunkStorageMetadata(offsets=torch.Size((self.offset, 0)), sizes=self.rows.shape)]

    def __create_write_items__(self, fqn, object):
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunk.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=chunk, properties=TensorProperties.create_from_tensor(self.weight), size=self.shape
                ),
            )
            for chunk in self.__create_chunk_list__()
        ]

    def __get_tensor_shard__(self, index):
        if index.offset != torch.Size((self.offset, 0)):
            raise ValueError(f"Unexpected Engram checkpoint offset: {index.offset}")
        return self.rows


def dequantize_engram_weight(weight, scale, *, row_block):
    """Decode official row-MX embeddings or block-MX gate projection weights."""
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("Engram quantized weights and scales must be matrices")
    expected = ((weight.shape[0] + row_block - 1) // row_block, (weight.shape[1] + 31) // 32)
    if tuple(scale.shape) != expected:
        raise ValueError(f"Engram scale shape {tuple(scale.shape)} does not match {expected}")
    result = torch.empty(weight.shape, dtype=torch.float32, device=weight.device)
    for start in range(0, weight.shape[0], 4096):
        end = min(start + 4096, weight.shape[0])
        scales = scale[start // row_block : (end + row_block - 1) // row_block].float()
        scales = scales.repeat_interleave(row_block, 0).repeat_interleave(32, 1)
        result[start:end] = weight[start:end].float() * scales[: end - start, : weight.shape[1]]
    return result


class EngramHuggingFaceStorageReader(QuantizedHuggingFaceStorageReader):
    """Extend the upstream reader with Engram's official `.scale` tensors."""

    def __init__(self, path, *, table_shapes, from_quantized):
        super().__init__(path, target_dtype=torch.float32, thread_count=4)
        self.table_shapes = table_shapes
        self.from_quantized = from_quantized

    def read_metadata(self):
        metadata = (
            self._read_quantized_metadata() if self.from_quantized else HuggingFaceStorageReader.read_metadata(self)
        )
        for name, shape in self.table_shapes.items():
            stored = metadata.state_dict_metadata.get(name)
            if stored is not None and (not isinstance(stored, TensorStorageMetadata) or tuple(stored.size) != shape):
                raise ValueError(f"HF Engram table {name} metadata does not match expected shape {shape}")
        return metadata

    def _read_quantized_metadata(self):
        # Official HF checkpoints contain whole tensors, with scales as sidecars.
        # Do not expose the sidecars to DCP: safetensors 0.8's Python dtype
        # lookup lacks E8M0 even though safe_open can read it correctly.
        self._load_quantization_metadata()
        tensors, storage = {}, {}
        for filename in sorted(set(self._weight_map.values())):
            path = str(Path(self.path) / filename)
            with safe_open(path, framework="pt", device="cpu") as file:
                for name in file.keys():  # noqa: SIM118 -- safe_open is not iterable
                    if name.endswith(".scale"):
                        continue
                    tensor = file.get_slice(name)
                    shape = torch.Size(tensor.get_shape())
                    dtype = _getdtype(tensor.get_dtype())
                    offsets = torch.Size([0] * len(shape))
                    tensors[name] = TensorStorageMetadata(
                        properties=TensorProperties(dtype=dtype),
                        size=shape,
                        chunks=[ChunkStorageMetadata(offsets=offsets, sizes=shape)],
                    )
                    storage[MetadataIndex(name, offsets)] = _HFStorageInfo(relative_path=path, shape=shape, dtype=dtype)
                    self._tensor_full_shapes[name] = shape
        return Metadata(
            state_dict_metadata=tensors, storage_data=storage, storage_meta=StorageMeta(load_id=self.load_id)
        )

    def _build_weight_scale_mapping(self, weight_map):
        super()._build_weight_scale_mapping(weight_map)
        for name in weight_map:
            if name.endswith((".engram.embed.scale", ".engram.wkv.scale")):
                weight_name = name.removesuffix("scale") + "weight"
                if weight_name in weight_map:
                    self._weight_scale_mapping[weight_name] = name

    def _process_read_request(self, f, req, planner):
        name = req.storage_index.fqn
        scale = name.removesuffix("weight") + "scale"
        if ".engram." in name and f.get_slice(name).get_dtype() == "F8_E4M3" and name not in self._weight_scale_mapping:
            raise ValueError(f"Quantized Engram weight {name} requires from_quantized=True and its scale tensor")
        if scale in self._weight_map and name not in self._weight_scale_mapping:
            raise NotImplementedError(
                f"The upstream HF reader does not support the quantization format of {name}; "
                "convert non-Engram weights to a supported HF format first."
            )
        return super()._process_read_request(f, req, planner)

    def _read_quantized_tensor_with_block_alignment(self, req, safetensor_file):
        name = req.storage_index.fqn
        if not name.endswith((".engram.embed.weight", ".engram.wkv.weight")):
            return super()._read_quantized_tensor_with_block_alignment(req, safetensor_file)
        row_block = 1 if name.endswith(".embed.weight") else 32
        row, col = req.storage_offsets
        height, width = req.lengths
        r0, c0 = row // row_block * row_block, col // 32 * 32
        r1 = (row + height + row_block - 1) // row_block * row_block
        c1 = (col + width + 31) // 32 * 32
        weight = safetensor_file.get_slice(name)[r0:r1, c0:c1]
        scale_name = self._weight_scale_mapping[name]
        with safe_open(str(Path(self.path) / self._weight_map[scale_name]), framework="pt", device="cpu") as sf:
            scale = sf.get_slice(scale_name)[r0 // row_block : (r1 + row_block - 1) // row_block, c0 // 32 : c1 // 32]
        values = dequantize_engram_weight(weight, scale, row_block=row_block)
        return values[row - r0 : row - r0 + height, col - c0 : col - c0 + width]
