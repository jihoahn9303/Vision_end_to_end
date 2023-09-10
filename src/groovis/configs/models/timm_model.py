from hydra.core.config_store import ConfigStore
from timm import create_model

from src.groovis.configs import full_builds

TimmModeConfig = full_builds(create_model, num_classes=0)

SmallViTTimmModelConfig = TimmModeConfig(model_name="vit_small_patch16_224")
BaseViTTimmModelConfig = TimmModeConfig(model_name="vit_base_patch16_224")
LargeViTTimmModelConfig = TimmModeConfig(model_name="vit_large_patch16_224")


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="architecture", name="vit_small_tim", node=SmallViTTimmModelConfig)
    cs.store(group="architecture", name="vit_base_tim", node=BaseViTTimmModelConfig)
    cs.store(group="architecture", name="vit_large_tim", node=LargeViTTimmModelConfig)
