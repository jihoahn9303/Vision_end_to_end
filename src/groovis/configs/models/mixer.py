from hydra.core.config_store import ConfigStore

from src.groovis.configs import full_builds
from src.groovis.configs.models import ArchitectureConfig, EmbedDim, PatchEmbedConfig
from src.groovis.models.mixer import Mixer

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
            backbone=MixerConfig(embed_dim=EmbedDim.SMALL.value),
        ),
    )
    cs.store(
        group="architecture",
        name="mixer_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=MixerConfig(embed_dim=EmbedDim.BASE.value),
        ),
    )
    cs.store(
        group="architecture",
        name="mixer_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=MixerConfig(embed_dim=EmbedDim.LARGE.value),
        ),
    )
