from hydra_zen import instantiate
from hydra_zen.typing import Partial
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.groovis.configs import Config
from src.groovis.module import Vision

# <TODO List for Training and Validation>
# TODO: Learning Rate Scheduling
# TODO: Control epochs
# TODO: Gradient Clipping
# TODO: Validation

# <TODO List for Callback>
# TODO: Checkpointing (Optimizer and Scheduler's state as well)
# TODO: Learning Rate Monitoring
# TODO: Early Stopping

# <TODO List for Logging>
# TODO: Custom Logging (Loguru)
# TODO: Wandb Logging
# TODO: Wandb Configuration Update
# TODO: Control Log Frequency (Aggregated by MeanMetric)
# TODO: Wandb Watch


def train(config: Config):
    experiment: Config = instantiate(config)

    # load data
    datamodule: LightningDataModule = experiment.datamodule

    # define loss function and metric
    loss_fn: nn.Module = experiment.loss

    # initialize architecture
    architecture: nn.Module = experiment.architecture

    # set optimizer and scheduler
    optimizer: Partial[Optimizer] = experiment.optimizer
    scheduler: Partial[_LRScheduler] = experiment.scheduler

    # initialize model
    model = Vision(
        architecture=architecture,
        loss_fn=loss_fn,
        optmizer=optimizer,
        scheduler=scheduler,
    )

    # set trainer
    trainer: Trainer = experiment.trainer

    # set logger
    logger: WandbLogger = trainer.logger

    logger.watch(
        model=architecture, log="all", log_freq=config.trainer.log_every_n_steps
    )

    if trainer.is_global_zero:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.config.update(OmegaConf.to_container(config))

    trainer.fit(model=model, datamodule=datamodule)
