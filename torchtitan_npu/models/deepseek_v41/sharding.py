"""V4.1-only sharding extensions layered on top of the V4 base policy."""

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.protocols.sharding import ShardingConfig

_DENSE_PARAM_REP = dense_param_placement(tp=spmd.R)


def set_deepseek_v41_sharding_extensions(config) -> None:
    """Assign layouts for V4.1-only vision and VL-router parameters."""
    if config.image_marker_embeddings is not None:
        config.image_marker_embeddings.sharding_config = ShardingConfig(
            state_shardings=dict.fromkeys(
                ("image_start", "image_newline", "image_end"),
                _DENSE_PARAM_REP,
            )
        )

    for layer_cfg in config.layers:
        layer_cfg.moe.router.sharding_config = ShardingConfig(
            state_shardings={"bias_vl": _DENSE_PARAM_REP}
        )
