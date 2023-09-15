import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()  # load environmental variables in .env file

import hydra
import torch
from hydra_zen import instantiate
from hydra_zen.typing import Partial
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import wandb
from src.groovis.configs import Config, register_configs
from src.groovis.module import Vision
from src.groovis.utils import image_path_to_tensor_inference


# import config automatically by hydra.main
@hydra.main(config_name="default", version_base="1.1")
def main(config: Config):
    loss_fn: nn.Module = instantiate(config.loss)
    architecture: nn.Module = instantiate(config.architecture)
    optimizer: Partial[Optimizer] = instantiate(config.optimizer)
    scheduler: Partial[_LRScheduler] = instantiate(config.scheduler)

    # download model arfifact from wandb server
    run = wandb.init()
    artifact = run.use_artifact("jihoahn9303/groovis/model-udab0aj6:v2", type="model")
    artifact_path = artifact.download()
    checkpoint_path = Path(artifact_path) / "model.ckpt"

    # initialize model
    model = Vision.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # load images
    image_tiger_1 = image_path_to_tensor_inference(
        "/workspaces/vision/datas/test/tiger_1.jpg"
    )
    image_tiger_2 = image_path_to_tensor_inference(
        "/workspaces/vision/datas/test/tiger_2.jpg"
    )
    image_dog = image_path_to_tensor_inference("/workspaces/vision/datas/test/dog.jpg")

    # get representation for images from trained model
    tiger_1: torch.Tensor = model(image_tiger_1)
    tiger_2: torch.Tensor = model(image_tiger_2)
    dog: torch.Tensor = model(image_dog)

    tiger_1.div_(tiger_1.norm())
    tiger_2.div_(tiger_2.norm())
    dog.div_(dog.norm())

    sim_tiger_tiger = (tiger_2 * tiger_1).sum()
    sim_tiger_dog_1 = (tiger_1 * dog).sum()
    sim_tiger_dog_2 = (tiger_2 * dog).sum()

    # calculate quality for representation
    quality = -(sim_tiger_dog_1 + sim_tiger_dog_2) / 2 + sim_tiger_tiger

    print(f"{sim_tiger_tiger=}")
    print(f"{sim_tiger_dog_1=}")
    print(f"{sim_tiger_dog_2=}")

    print(quality)


if __name__ == "__main__":
    register_configs()
    main()
