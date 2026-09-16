# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .attention import (
    DeepSeekV41Attention,
    V41AttentionContext,
    V41CompressionSpec,
    build_v41_compression_spec,
)
from .config import (
    V41_CANDIDATE_SOURCE_LAYER,
    V41_COMPRESS_RATIOS,
    V41_FULL_COMPRESS_RATIOS,
    V41_FULL_INDEX_SOURCE_LAYERS,
    V41_FULL_LAYER_IDS,
    V41_INDEX_SOURCE_LAYERS,
    V41_KV_SOURCE_LAYERS,
    V41_LAYER_IDS,
    DeepSeekV41CropConfig,
    DeepSeekV41FullLayerConfig,
    DeepSeekV41FullScaleProfile,
    EngramArgs,
)
from .config_registry import (
    deepseek_v41_flash_30layers_16experts_vision,
    deepseek_v41_flash_40layers_16experts_vision,
)
from .model import V41Model
from .model_registry import (
    deepseek_v41_flash_30layers_16experts_vision_config,
    deepseek_v41_flash_40layers_16experts_vision_config,
    model_registry,
)
from .state_dict_adapter import DeepSeekV41StateDictAdapter

__all__ = [
    "V41_CANDIDATE_SOURCE_LAYER",
    "V41_COMPRESS_RATIOS",
    "V41_FULL_COMPRESS_RATIOS",
    "V41_FULL_INDEX_SOURCE_LAYERS",
    "V41_FULL_LAYER_IDS",
    "V41_INDEX_SOURCE_LAYERS",
    "V41_KV_SOURCE_LAYERS",
    "V41_LAYER_IDS",
    "DeepSeekV41Attention",
    "DeepSeekV41CropConfig",
    "DeepSeekV41FullLayerConfig",
    "DeepSeekV41FullScaleProfile",
    "DeepSeekV41StateDictAdapter",
    "EngramArgs",
    "V41AttentionContext",
    "V41CompressionSpec",
    "V41Model",
    "build_v41_compression_spec",
    "deepseek_v41_flash_30layers_16experts_vision",
    "deepseek_v41_flash_30layers_16experts_vision_config",
    "deepseek_v41_flash_40layers_16experts_vision",
    "deepseek_v41_flash_40layers_16experts_vision_config",
    "model_registry",
]
