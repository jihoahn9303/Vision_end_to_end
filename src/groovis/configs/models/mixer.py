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
from src.groovis.models.mixer import Mixer, MixerBlock

MixerBlockConfig = partial_builds(
    MixerBlock,
    expansion_factor=4,
)
MixerConfig = full_builds(Mixer)


def _register_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="architecture",
        name="mixer_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=MixerConfig(
                block=MixerBlockConfig(
                    embed_dim=EmbedDim.SMALL.value,
                    act_layer=GELUConfig,
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
            backbone=MixerConfig(
                block=MixerBlockConfig(
                    embed_dim=EmbedDim.BASE.value,
                    act_layer=GELUConfig,
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
            backbone=MixerConfig(
                block=MixerBlockConfig(
                    embed_dim=EmbedDim.LARGE.value,
                    act_layer=GELUConfig,
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.LARGE.value),
                depth=Depth.LARGE.value,
            ),
        ),
    )
