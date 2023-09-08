from hydra.core.config_store import ConfigStore
from timm import create_model

from src.groovis.configs import full_builds
from src.groovis.models.architectures import Architecture

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
SmallArchitectureConfig = ArchitectureConfig(embed_dim=384)
BaseArchitectureConfig = ArchitectureConfig(embed_dim=768)
LargeArchitectureConfig = ArchitectureConfig(embed_dim=1024)

TimmModeConfig = full_builds(create_model, num_classes=0)
SmallViTTimmModelConfig = TimmModeConfig(model_name="vit_small_tim")
BaseViTTimmModelConfig = TimmModeConfig(model_name="vit_base_tim")
LargeViTTimmModelConfig = TimmModeConfig(model_name="vit_large_tim")


def _register_configs():
    cs = ConfigStore.instance()
    cs.store(group="architecture", name="small", node=SmallArchitectureConfig)
    cs.store(group="architecture", name="base", node=BaseArchitectureConfig)
    cs.store(group="architecture", name="large", node=LargeArchitectureConfig)
    cs.store(group="architecture", name="vit_small_tim", node=SmallViTTimmModelConfig)
    cs.store(group="architecture", name="vit_base_tim", node=BaseViTTimmModelConfig)
    cs.store(group="architecture", name="vit_large_tim", node=LargeViTTimmModelConfig)
