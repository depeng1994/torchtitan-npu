# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Round-trip checks for the Engram tokenizer-compression map.

The map is generated offline and consumed at model build time. Nothing tied
the two ends together, so a change to the normalization rules or to the file
format could only surface as wrong row IDs during a real run.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from scripts.generate_engram_token_map import build_token_id_map, normalize_token
from torchtitan_npu.models.deepseek_v41.config import EngramArgs
from torchtitan_npu.models.deepseek_v41.engram_config import _make_engram_configs

pytestmark = pytest.mark.tooling

_TOKENIZER = Path(__file__).parents[4] / "tests" / "assets" / "deepseek_v3" / "tokenizer.json"


def _generated_map():
    return build_token_id_map(Tokenizer.from_file(str(_TOKENIZER)))


def test_normalization_collapses_case_accents_and_whitespace():
    assert normalize_token("Hello") == normalize_token("hello")
    assert normalize_token("café") == normalize_token("cafe")
    assert normalize_token("a\t b") == "a b"
    # A token that normalizes to a single space keeps it; others are stripped.
    assert normalize_token(" ") == " "
    assert normalize_token("  word  ") == "word"
    assert normalize_token("  Apple\n") == "apple"
    assert normalize_token("\t\n") == " "
    # Normalizing to nothing falls back to the original text.
    assert normalize_token("\u200b") == "\u200b"


def test_generated_map_compresses_the_test_tokenizer():
    mapping = _generated_map()
    tokenizer = Tokenizer.from_file(str(_TOKENIZER))

    assert mapping.shape == (tokenizer.get_vocab_size(with_added_tokens=True),)
    assert mapping.dtype == np.int64
    assert mapping.min() >= 0
    compressed = int(mapping.max()) + 1
    # Compression is the point: distinct token IDs must share canonical IDs.
    assert compressed < mapping.shape[0]
    # Compressed IDs are dense, so the table has no unused rows.
    assert sorted(set(mapping.tolist())) == list(range(compressed))


def test_generated_map_loads_into_the_table(tmp_path):
    mapping = _generated_map()
    compressed = int(mapping.max()) + 1
    map_path = tmp_path / "engram_token_id_map.npy"
    np.save(map_path, mapping)

    def build(expected_compressed):
        args = EngramArgs(
            layer_ids=(0,),
            vocab_size_per_ngram=(16, 16),
            n_embed_per_ngram=16,
            num_heads_per_ngram=2,
            token_id_map_path=str(map_path),
            require_token_id_map=True,
            compressed_vocab_size=expected_compressed,
        )
        configs = _make_engram_configs(hidden_size=8, hc_mult=1, vocab_size=mapping.shape[0], engram=args)
        table = configs[0].table.build()
        table.init_states()
        return table

    table = build(compressed)
    assert torch.equal(table.token_id_map.cpu(), torch.from_numpy(mapping))

    with pytest.raises(ValueError, match="Regenerate the map"):
        build(compressed + 1)


def test_a_required_map_must_be_configured():
    args = EngramArgs(
        layer_ids=(0,),
        vocab_size_per_ngram=(16, 16),
        n_embed_per_ngram=16,
        num_heads_per_ngram=2,
        require_token_id_map=True,
    )
    configs = _make_engram_configs(hidden_size=8, hc_mult=1, vocab_size=32, engram=args)
    table = configs[0].table.build()
    with pytest.raises(ValueError, match="token_id_map_path is not set"):
        table.init_states()
