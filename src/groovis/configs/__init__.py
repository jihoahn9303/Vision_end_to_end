import pkgutil
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_zen import make_custom_builds_fn
from omegaconf import MISSING

defaults = [
    "_self_",
    {"architecture": "base"},
    {"loss": "nt_xent_medium"},
    {"datamodule": "imagenet"},
    {"datamodule/dataset": "imagenette"},
    {"datamodule/dataset/transforms": "relaxed"},
    {"datamodule/dataloader": "base"},
    {"trainer": "auto"},
    {"trainer/callbacks": "default"},
    {"optimizer": "adam"},
    {"scheduler": "onecycle"},
]


@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    architecture: Any = MISSING
    loss: Any = MISSING
    datamodule: Any = MISSING
    trainer: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING


full_builds = make_custom_builds_fn(
    # Using default hyperparameter
    populate_full_signature=True
)
partial_builds = make_custom_builds_fn(populate_full_signature=True, zen_partial=True)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="default", node=Config)

    for module_info in pkgutil.walk_packages(__path__):
        name = module_info.name
        module_finder = module_info.module_finder

        module = module_finder.find_module(name).load_module(name)
        if hasattr(module, "_register_configs"):
            module._register_configs()
