# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["DeepSeekV4PEFTCheckpointManager", "select_adapter_only_state"]

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import MODEL
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.tools.logging import logger

from torchtitan_npu.extensions.components.checkpoint import CheckpointManager as NPUCheckpointManager


def select_adapter_only_state(
    flattened_state: dict[str, Any],
    non_model_keys: set[str],
    *,
    model_buffer_keys: set[str] | None = None,
    model_trainable_keys: set[str] | None = None,
) -> dict[str, Any]:
    retained_keys = non_model_keys | (model_buffer_keys or set()) | (model_trainable_keys or set())
    return {key: value for key, value in flattened_state.items() if key in retained_keys or "lora_" in key}


class DeepSeekV4PEFTCheckpointManager(NPUCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUCheckpointManager.Config):
        last_save_in_peft: bool = True
        periodic_save_adapter_only: bool = True
        save_training_state: bool = False

    def __init__(self, config: Config, **kwargs) -> None:
        if config.enable:
            if config.last_save_in_hf:
                raise ValueError("last_save_in_hf cannot preserve LoRA adapters; use PEFT export or native DCP")
            if config.last_save_in_peft:
                from .state_dict_adapter import DeepSeekV4StateDictAdapter

                adapter = kwargs.get("sd_adapter")
                if not isinstance(adapter, DeepSeekV4StateDictAdapter):
                    raise TypeError("DeepSeekV4 PEFT export requires DeepSeekV4StateDictAdapter")
                adapter.peft_adapter_config()
        super().__init__(config, **kwargs)
        if not config.enable:
            return
        self.last_save_in_peft = config.last_save_in_peft
        self.periodic_save_adapter_only = config.periodic_save_adapter_only
        self.save_training_state = config.save_training_state
        if self.periodic_save_adapter_only and not self.initial_load_path:
            raise ValueError("periodic_save_adapter_only requires a fixed initial_load_path for the frozen base")

    @staticmethod
    def _finalize_peft_directory(checkpoint_id: str, adapter_config: dict) -> None:
        consolidated = [
            name for name in os.listdir(checkpoint_id) if name.startswith("model-") and name.endswith(".safetensors")
        ]
        if len(consolidated) != 1:
            raise RuntimeError(
                f"Expected one consolidated PEFT safetensors file in {checkpoint_id}, got {consolidated}"
            )
        os.replace(
            os.path.join(checkpoint_id, consolidated[0]), os.path.join(checkpoint_id, "adapter_model.safetensors")
        )
        index_path = os.path.join(checkpoint_id, "model.safetensors.index.json")
        if os.path.exists(index_path):
            os.remove(index_path)
        sharded_path = os.path.join(checkpoint_id, "sharded")
        if os.path.isdir(sharded_path):
            shutil.rmtree(sharded_path)
        with open(os.path.join(checkpoint_id, "adapter_config.json"), "w", encoding="utf-8") as output:
            json.dump(adapter_config, output, indent=2, sort_keys=True)
            output.write("\n")

    @classmethod
    def finalize_peft_checkpoint(cls, checkpoint_id: str, adapter_config: dict) -> None:
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        status: list[str | None] = [None]
        if rank == 0:
            try:
                cls._finalize_peft_directory(checkpoint_id, adapter_config)
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        if distributed:
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"PEFT checkpoint finalization failed: {status[0]}")

    def dcp_load(self, state_dict, checkpoint_id, from_hf=False, from_quantized=False):
        if not from_hf and checkpoint_id == self.initial_load_path:
            state_dict = {key: value for key, value in state_dict.items() if "lora_" not in key}
        elif not from_hf:
            saved_keys = dcp.FileSystemReader(checkpoint_id).read_metadata().state_dict_metadata.keys()
            model_state = self.states[MODEL].state_dict()
            model_buffer_keys = self._model_buffer_keys()
            has_base = all(
                key in saved_keys for key in model_state if "lora_" not in key and key not in model_buffer_keys
            )
            has_adapters = any("lora_" in key for key in saved_keys)
            if not has_adapters:
                state_dict = {key: value for key, value in state_dict.items() if "lora_" not in key}
            if has_base or not has_adapters:
                return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)
            if not self.initial_load_path:
                raise ValueError("Loading an adapter-only DCP requires initial_load_path for the frozen base")
            self.dcp_load(
                model_state, self.initial_load_path, self.initial_load_in_hf, self.initial_load_in_hf_quantized
            )
            non_model_keys = {
                key
                for key in self.states
                if key != MODEL and any(saved == key or saved.startswith(f"{key}.") for saved in saved_keys)
            }
            if not non_model_keys:
                logger.warning(
                    "Loading LoRA weights only: optimizer, scheduler, dataloader and training step are not restored."
                )
            state_dict = select_adapter_only_state(
                {**state_dict, **self.states[MODEL].state_dict()},
                non_model_keys,
                model_buffer_keys=model_buffer_keys.intersection(saved_keys),
                model_trainable_keys=set(model_state).intersection(saved_keys),
            )
        return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)

    def _model_buffer_keys(self) -> set[str]:
        return {
            canonical_fqn(name)
            for model in self.states[MODEL].model
            for name, _ in model.named_buffers(remove_duplicate=False)
        }

    def _ensure_peft_exportable(self) -> None:
        unsupported = sorted(
            name
            for model in self.states[MODEL].model
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad and "lora_" not in name
        )

        if dist.is_available() and dist.is_initialized():
            gathered: list[list[str]] = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, unsupported)
            unsupported = sorted({name for names in gathered for name in names})
        if unsupported:
            preview = ", ".join(unsupported[:8])
            suffix = "..." if len(unsupported) > 8 else ""
            raise ValueError(
                "PEFT LoRA export cannot preserve trainable non-LoRA parameters: "
                f"{preview}{suffix}. Freeze them or save an adapter DCP instead."
            )

    def _flattened_model_states_sd(self, state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        flattened = super()._flattened_model_states_sd(state_dict)
        if not self.periodic_save_adapter_only or state_dict is not None:
            return flattened
        non_model_keys = {key for key in self.states if key != MODEL} if self.save_training_state else set()
        trainable_keys = {
            canonical_fqn(name)
            for model in self.states[MODEL].model
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        }
        selected = select_adapter_only_state(
            flattened, non_model_keys, model_buffer_keys=self._model_buffer_keys(), model_trainable_keys=trainable_keys
        )
        if not any("lora_" in key for key in selected):
            raise ValueError("No LoRA adapter tensors were found for interval checkpointing")
        return selected

    def _save_last_step(self, curr_step: int) -> None:
        if not self.last_save_in_peft:
            super()._save_last_step(curr_step)
            return
        from .state_dict_adapter import DeepSeekV4StateDictAdapter

        if not isinstance(self.sd_adapter, DeepSeekV4StateDictAdapter):
            raise TypeError("DeepSeekV4 PEFT export requires DeepSeekV4StateDictAdapter")

        self._ensure_peft_exportable()
        state_dict = self.states[MODEL].state_dict()
        peft_state_dict = self.sd_adapter.to_peft(state_dict)
        adapter_config = self.sd_adapter.peft_adapter_config(base_model_name_or_path=self.initial_load_path)
        if self.export_dtype != torch.float32:
            peft_state_dict = {key: value.to(self.export_dtype) for key, value in peft_state_dict.items()}
        checkpoint_id = self._create_checkpoint_id(curr_step)
        logger.info("Saving PEFT LoRA adapter checkpoint at %s", checkpoint_id)
        writer = HuggingFaceStorageWriter(path=checkpoint_id, save_distributed=True, enable_consolidation=True)
        dcp.save(peft_state_dict, storage_writer=writer)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            if self.initial_load_path and not self.initial_load_in_hf:
                logger.warning(
                    "PEFT base path %s is a native checkpoint; external PEFT loaders require converted HF weights.",
                    self.initial_load_path,
                )
            elif not self.initial_load_path:
                logger.warning(
                    "PEFT base path defaults to hf_assets_path; set initial_load_path to the actual base weights."
                )
        self.finalize_peft_checkpoint(checkpoint_id, adapter_config)
