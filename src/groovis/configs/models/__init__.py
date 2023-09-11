from enum import IntEnum

from hydra.core.config_store import ConfigStore

from src.groovis.configs import full_builds
from src.groovis.models import Architecture
from src.groovis.models.components.patch_embed import PatchEmbed


class Depth(IntEnum):
    SMALL = 8
    BASE = 12
    LARGE = 24


class EmbedDim(IntEnum):
    SMALL = 384
    BASE = 768
    LARGE = 1024


# Hydra version
# def import_path(x: Any):
#     return f"{x.__module__}.{x.__name__}"

# @dataclass
# class TimmModelConfig:
#     _target_: str = import_path(create_model)
#     model_name: str = MISSING
#     num_classes: int = 0


# @dataclass
# class SmallViTTimmModelConfig(TimmModelConfig):
#     model_name: str = "vit_small_patch16_224"


# @dataclass
# class BaseViTTimmModelConfig(TimmModelConfig):
#     model_name: str = "vit_base_patch16_224"


# @dataclass
# class LargeViTTimmModelConfig(TimmModelConfig):
#     model_name: str = "vit_large_patch16_224"


# @dataclass
# class ArchitectureConfig:
#     _target_: str = import_path(Architecture)
#     patch_size: int = 16
#     channels: int = 3
#     embed_dim: int = MISSING


# @dataclass
# class SmallArchitectureConfig(ArchitectureConfig):
#     embed_dim: int = 384


# @dataclass
# class BaseArchitectureConfig(ArchitectureConfig):
#     embed_dim: int = 786


# @dataclass
# class LargeArchitectureConfig(ArchitectureConfig):
#     embed_dim: int = 1024


# Hydra-zen Version
# return dataclasses type
ArchitectureConfig = full_builds(Architecture)
PatchEmbedConfig = full_builds(PatchEmbed)


def _register_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="architecture",
        name="simple_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=None,
        ),
    )
    cs.store(
        group="architecture",
        name="simple_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=None,
        ),
    )
    cs.store(
        group="architecture",
        name="simple_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=None,
        ),
    )
