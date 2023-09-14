from hydra.core.config_store import ConfigStore

from src.groovis.configs import full_builds, partial_builds
from src.groovis.configs.models import (
    ArchitectureConfig,
    Depth,
    EmbedDim,
    PatchEmbedConfig,
)
from src.groovis.configs.models.components.act_layer import GELUConfig
from src.groovis.configs.models.components.layer_norm import PreNormConfig
from src.groovis.models.vit import AlternatingBackbone, CrossTokenMixerBlock, MLPBlock

MLPBlockConfig = partial_builds(
    MLPBlock,
    expansion_factor=4,
    act_layer=GELUConfig,
)
CrossTokenMixerConfig = partial_builds(
    CrossTokenMixerBlock,
    expansion_factor=0.5,
    act_layer=GELUConfig,
)
AlternatingBackboneConfig = full_builds(AlternatingBackbone)


def _register_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="architecture",
        name="mixer_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=AlternatingBackboneConfig(
                per_location_block=MLPBlockConfig(
                    embed_dim=EmbedDim.SMALL.value,
                ),
                cross_location_block=CrossTokenMixerConfig(
                    seq_len=14 * 14,
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.SMALL.value),
                depth=Depth.SMALL.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="mixer_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=AlternatingBackboneConfig(
                per_location_block=MLPBlockConfig(
                    embed_dim=EmbedDim.BASE.value,
                ),
                cross_location_block=CrossTokenMixerConfig(
                    seq_len=14 * 14,
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.BASE.value),
                depth=Depth.BASE.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="mixer_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=AlternatingBackboneConfig(
                per_location_block=MLPBlockConfig(
                    embed_dim=EmbedDim.LARGE.value,
                ),
                cross_location_block=CrossTokenMixerConfig(
                    seq_len=14 * 14,
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.LARGE.value),
                depth=Depth.LARGE.value,
            ),
        ),
    )
