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
from src.groovis.models.vit import (
    AlternatingBackbone,
    FusedTransformerBlock,
    HomogeneousBackbone,
    MLPBlock,
    SelfAttention,
)

MLPBlockConfig = partial_builds(
    MLPBlock,
    expansion_factor=4,
    act_layer=GELUConfig,
)
FusedTransformerBlockConfig = partial_builds(
    FusedTransformerBlock,
    expansion_factor=4,
    act_layer=GELUConfig,
)
SelfAttentionConfig = partial_builds(SelfAttention)
AlternatingBackboneConfig = full_builds(AlternatingBackbone)
HomogeneousBackboneConfig = full_builds(HomogeneousBackbone)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="architecture",
        name="vit_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=AlternatingBackboneConfig(
                per_location_block=MLPBlockConfig(
                    embed_dim=EmbedDim.SMALL.value,
                ),
                cross_location_block=SelfAttentionConfig(
                    embed_dim=EmbedDim.SMALL.value, num_heads=6
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.SMALL.value),
                depth=Depth.SMALL.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="vit_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=AlternatingBackboneConfig(
                per_location_block=MLPBlockConfig(
                    embed_dim=EmbedDim.BASE.value,
                ),
                cross_location_block=SelfAttentionConfig(
                    embed_dim=EmbedDim.BASE.value, num_heads=12
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.BASE.value),
                depth=Depth.BASE.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="vit_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=AlternatingBackboneConfig(
                per_location_block=MLPBlockConfig(
                    embed_dim=EmbedDim.LARGE.value,
                ),
                cross_location_block=SelfAttentionConfig(
                    embed_dim=EmbedDim.LARGE.value, num_heads=16
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.LARGE.value),
                depth=Depth.LARGE.value,
            ),
        ),
    )

    cs.store(
        group="architecture",
        name="parallel_vit__small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=HomogeneousBackboneConfig(
                block=FusedTransformerBlockConfig(
                    embed_dim=EmbedDim.SMALL.value, num_heads=6
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.SMALL.value),
                depth=Depth.SMALL.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="parallel_vit_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=HomogeneousBackboneConfig(
                block=FusedTransformerBlockConfig(
                    embed_dim=EmbedDim.BASE.value, num_heads=12
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.BASE.value),
                depth=Depth.BASE.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="parallel_vit_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=HomogeneousBackboneConfig(
                block=FusedTransformerBlockConfig(
                    embed_dim=EmbedDim.LARGE.value, num_heads=16
                ),
                norm=PreNormConfig(embed_dim=EmbedDim.LARGE.value),
                depth=Depth.LARGE.value,
            ),
        ),
    )
