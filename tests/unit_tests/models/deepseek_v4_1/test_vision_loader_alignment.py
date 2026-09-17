# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepSeek V4.1 vision loader document alignment.

Every row ``_SyntheticVisionDataset`` emits is a single document
(``positions == arange(seq_len)``, one ``positions == 0``), so the row length has
to be a multiple of the model's pooling ratio: otherwise a compressed-attention
pool straddles the document edge silently.  A row cannot be padded after it has
been cut, so the requirement is validated when the dataset and the loader are
built, and the value is derived from the model's compression ratios.
"""

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1 import model_registry as v41_model_registry
from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    _document_alignment,
    deepseek_v4_1_debugmodel,
)
from torchtitan_npu.models.deepseek_v4_1.vision_loader import (
    DeepSeekV41SyntheticVisionDataLoader,
    _SyntheticVisionDataset,
)

_SEQ_LEN = 512
_VOCAB_SIZE = 129280
_ALIGNMENT = 2


def _dataset(**overrides) -> _SyntheticVisionDataset:
    params = dict(vocab_size=_VOCAB_SIZE, seq_len=_SEQ_LEN, patch_count=64, span_start=8)
    params.update(overrides)
    return _SyntheticVisionDataset(**params)


def _loader(**config_overrides):
    config = DeepSeekV41SyntheticVisionDataLoader.Config(**config_overrides)
    return DeepSeekV41SyntheticVisionDataLoader(
        config, dp_world_size=1, dp_rank=0, seq_len=_SEQ_LEN, local_batch_size=1
    )


def test_rows_are_single_documents_on_the_pooling_grid():
    dataset = _dataset(document_alignment=_ALIGNMENT)
    assert dataset.document_alignment == _ALIGNMENT

    input_dict, labels = next(iter(dataset))
    positions = input_dict["positions"]

    assert positions.numel() == _SEQ_LEN
    torch.testing.assert_close(positions, torch.arange(_SEQ_LEN), rtol=0, atol=0)
    assert int((positions == 0).sum()) == 1
    assert input_dict["input"].numel() == _SEQ_LEN
    assert labels.numel() == _SEQ_LEN


def test_unaligned_seq_len_is_rejected():
    with pytest.raises(ValueError, match="seq_len"):
        _dataset(seq_len=511, document_alignment=_ALIGNMENT)
    with pytest.raises(ValueError, match="document_alignment"):
        _dataset(document_alignment=0)


def test_alignment_only_validates_and_leaves_the_row_unchanged():
    """Requiring the alignment must not alter the emitted row."""
    aligned = _dataset(document_alignment=_ALIGNMENT)
    default = _dataset()
    assert default.document_alignment == 1

    aligned_input, aligned_labels = next(iter(aligned))
    default_input, default_labels = next(iter(default))
    for key in aligned_input:
        torch.testing.assert_close(aligned_input[key], default_input[key], rtol=0, atol=0)
    torch.testing.assert_close(aligned_labels, default_labels, rtol=0, atol=0)


def test_loader_passes_the_alignment_to_the_dataset():
    loader = _loader(document_alignment=_ALIGNMENT)
    assert loader.dataset.document_alignment == _ALIGNMENT


def test_loader_rejects_an_unaligned_seq_len():
    config = DeepSeekV41SyntheticVisionDataLoader.Config(document_alignment=_ALIGNMENT)
    with pytest.raises(ValueError, match="seq_len"):
        DeepSeekV41SyntheticVisionDataLoader(
            config, dp_world_size=1, dp_rank=0, seq_len=511, local_batch_size=1
        )


def test_alignment_is_derived_from_the_compression_ratios():
    spec = v41_model_registry("deepseek_v4_1_debugmodel")
    ratios = [ratio for ratio in spec.model.compress_ratios if ratio > 1]
    assert ratios and _document_alignment(spec) == max(ratios) == _ALIGNMENT

    # ... and reaches the registered recipe.
    config = deepseek_v4_1_debugmodel()
    assert config.dataloader.document_alignment == _ALIGNMENT
