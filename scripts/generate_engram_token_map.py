#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the tokenizer-compression lookup used by DeepSeek Engram."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, normalizers

# A private-use character, so a token that is exactly one space survives Strip()
# instead of collapsing to the empty string and merging with unrelated tokens.
_SPACE_SENTINEL = "\ue000"


@functools.cache
def _normalizer() -> normalizers.Normalizer:
    """The reference normalization pipeline.

    Built from ``tokenizers`` rather than reimplemented in Python: the two
    disagree in ways that silently change row IDs. ``str.lower()`` applies the
    Greek final-sigma rule that ``Lowercase`` does not, and filtering by
    combining class keeps the marks ``StripAccents`` removes whose canonical
    combining class is zero.
    """
    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), _SPACE_SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(_SPACE_SENTINEL, " "),
        ]
    )


def normalize_token(text: str) -> str:
    """Normalize one token's text; empty results fall back to the raw text."""
    normalized = _normalizer().normalize_str(text)
    return normalized or text


def build_token_id_map(tokenizer: Tokenizer) -> np.ndarray:
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    mapping = np.empty(vocab_size, dtype=np.int64)
    canonical_to_id: dict[str, int] = {}
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            canonical = tokenizer.id_to_token(token_id)
            if canonical is None:
                raise ValueError(f"Tokenizer has no token string for ID {token_id}.")
        else:
            canonical = normalize_token(text)
        compressed_id = canonical_to_id.setdefault(canonical, len(canonical_to_id))
        mapping[token_id] = compressed_id
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tokenizer",
        type=Path,
        help="HF tokenizer directory or tokenizer.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .npy path (default: TOKENIZER_DIR/engram_token_id_map.npy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer
    tokenizer_json = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Tokenizer JSON does not exist: {tokenizer_json}")
    output = args.output or tokenizer_json.parent / "engram_token_id_map.npy"
    if output.suffix != ".npy":
        raise ValueError(f"Engram token map output must use the .npy suffix, got {output}.")

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    mapping = build_token_id_map(tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, mapping, allow_pickle=False)
    print(f"Saved {mapping.size} token IDs -> {int(mapping.max()) + 1} canonical IDs to {output}")


if __name__ == "__main__":
    main()
