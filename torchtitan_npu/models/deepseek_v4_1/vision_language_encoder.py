# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 model-owned vision-language encoder."""

import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from torchtitan.components.tokenizer import HuggingFaceTokenizer

ThinkingMode = Literal["chat", "thinking"]
ReasoningEffort = Literal["low", "high", "max"] | int


class DeepSeekV41VisionLanguageEncoder:
    """Apply the model asset's official template and assistant supervision."""

    def __init__(
        self,
        encoding_module_path: str,
        thinking_mode: ThinkingMode = "chat",
        drop_thinking: bool = True,
        add_default_bos_token: bool = True,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"Unsupported thinking_mode: {thinking_mode!r}")
        if reasoning_effort not in (None, "low", "high", "max") and not (
            type(reasoning_effort) is int and 1 <= reasoning_effort <= 100
        ):
            raise ValueError(f"Unsupported reasoning_effort: {reasoning_effort!r}")
        spec = importlib.util.spec_from_file_location("encoding_dsv41", encoding_module_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load encoding module: {encoding_module_path}")
        self._encoding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._encoding)
        self.thinking_mode = thinking_mode
        self.drop_thinking = drop_thinking
        self.add_default_bos_token = add_default_bos_token
        self.reasoning_effort = reasoning_effort

    def _drop_thinking_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_user = self._encoding.find_last_user_index(messages)
        kept = []
        for index, message in enumerate(messages):
            role = message.get("role")
            if role in {"user", "system", "tool", "latest_reminder"} or index >= last_user:
                kept.append(message)
            elif role == "assistant":
                message = copy.copy(message)
                message.pop("reasoning_content", None)
                kept.append(message)
        return kept

    def encode_messages_with_assistant_mask(
        self,
        messages: list[dict[str, Any]],
        tokenizer: HuggingFaceTokenizer,
        *,
        context: list[dict[str, Any]] | None = None,
        thinking_mode: ThinkingMode | None = None,
        drop_thinking: bool | None = None,
        add_default_bos_token: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> tuple[str, list[int], list[bool], list[int], list[dict[str, Any]]]:
        """Return prompt, text tokens, assistant mask, image offsets, and images."""
        thinking_mode = thinking_mode or self.thinking_mode
        drop_thinking = self.drop_thinking if drop_thinking is None else drop_thinking
        add_bos = self.add_default_bos_token if add_default_bos_token is None else add_default_bos_token
        reasoning_effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort

        prompt, media = self._encoding.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            context=context or None,
            drop_thinking=drop_thinking,
            add_default_bos_token=add_bos,
            reasoning_effort=reasoning_effort,
            return_multi_modal_data=True,
        )
        images = media["images"]

        processed_context, _ = self._encoding.process_image_messages(context or [])
        processed_messages, _ = self._encoding.process_image_messages(messages)
        processed_messages = self._encoding.merge_tool_messages(processed_messages)
        processed_messages = self._encoding.sort_tool_results_by_call_order(processed_context + processed_messages)[
            len(processed_context) :
        ]
        if processed_context:
            processed_context = self._encoding.sort_tool_results_by_call_order(
                self._encoding.merge_tool_messages(processed_context)
            )
        full_messages = processed_context + processed_messages
        drop_thinking = drop_thinking and not any(message.get("tools") for message in full_messages)
        if thinking_mode == "thinking" and drop_thinking:
            full_messages = self._drop_thinking_messages(full_messages)
            context_len = len(self._drop_thinking_messages(processed_context))
        else:
            context_len = len(processed_context)

        rendered_prompt = self._encoding.bos_token if add_bos and not processed_context else ""
        assistant_spans = []
        for index in range(context_len, len(full_messages)):
            message = full_messages[index]
            rendered = self._encoding.render_message(
                index,
                full_messages,
                thinking_mode=thinking_mode,
                drop_thinking=drop_thinking,
                reasoning_effort=reasoning_effort,
            )
            start = len(rendered_prompt)
            rendered_prompt += rendered
            if message.get("role") == "assistant":
                assistant_spans.append((start, len(rendered_prompt)))
        if rendered_prompt != prompt:
            raise RuntimeError("assistant supervision rendering differs from the official V4.1 encoding")

        placeholder = self._encoding.IMAGE_PLACEHOLDER
        image_spans = []
        cursor = 0
        while (start := prompt.find(placeholder, cursor)) >= 0:
            image_spans.append((start, start + len(placeholder)))
            cursor = start + len(placeholder)
        if len(image_spans) != len(images):
            raise ValueError("Encoded image count does not match the prompt")

        cuts = sorted(image_spans + [(end, end) for _, end in assistant_spans])
        token_ids: list[int] = []
        assistant_mask: list[bool] = []
        image_positions: list[int] = []
        cursor = 0
        for start, end in [*cuts, (len(prompt), len(prompt))]:
            if start < cursor:
                continue
            encoded = tokenizer.tokenizer.encode(prompt[cursor:start])
            token_ids.extend(encoded.ids)
            assistant_mask.extend(
                any(
                    cursor + begin < span_end and cursor + finish > span_start
                    for span_start, span_end in assistant_spans
                )
                for begin, finish in encoded.offsets
            )
            if (
                start == end
                and any(span_end == end for _, span_end in assistant_spans)
                and token_ids
                and token_ids[-1] == tokenizer.eos_id
            ):
                assistant_mask[-1] = True
            if end > start:
                image_positions.append(len(token_ids))
            cursor = end
        return prompt, token_ids, assistant_mask, image_positions, images


@dataclass(kw_only=True, slots=True)
class DeepSeekV41VisionLanguageEncoderConfig:
    thinking_mode: ThinkingMode = "chat"
    drop_thinking: bool = True
    add_default_bos_token: bool = True
    reasoning_effort: ReasoningEffort | None = None

    def build(self, tokenizer_path: str) -> DeepSeekV41VisionLanguageEncoder:
        return DeepSeekV41VisionLanguageEncoder(
            str(Path(tokenizer_path) / "encoding" / "encoding.py"),
            thinking_mode=self.thinking_mode,
            drop_thinking=self.drop_thinking,
            add_default_bos_token=self.add_default_bos_token,
            reasoning_effort=self.reasoning_effort,
        )
