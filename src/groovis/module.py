import torch
from hydra_zen.typing import Partial
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MeanMetric

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"


# self-contained class (architecture + instruction for architecture)
class Vision(LightningModule):
    def __init__(
        self,
        architecture: nn.Module,
        loss_fn: nn.Module,
        optimizer: Partial[Optimizer],
        scheduler: Partial[_LRScheduler],
    ) -> None:
        super().__init__()

        self.architecture = architecture
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.architecture(images)

    def training_step(self, batch: list[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        images_1, images_2 = batch

        representations_1 = self(images_1)
        representations_2 = self(images_2)

        loss = self.loss_fn(representations_1, representations_2)
        self.train_loss(loss)  # self.train_loss.update(loss)
        self.log(
            name=TRAIN_LOSS,
            value=self.train_loss,  # compute metric and reset
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss  # return loss to update parameters

    def validation_step(self, batch: list[torch.Tensor], _batch_idx: int):
        images_1, images_2 = batch

        representations_1 = self(images_1)
        representations_2 = self(images_2)

        loss = self.loss_fn(representations_1, representations_2)
        self.val_loss(loss)
        self.log(
            name=VAL_LOSS,
            value=self.val_loss,
            on_step=False,
            on_epoch=True,
            logger=True,  # logs to other loggers(ex: Tensorboard, wandb, loguru, ...)
        )

    def configure_optimizers(self):
        # set optimizer
        optimizer = self.optimizer(params=self.parameters())

        # set scheduler
        scheduler = self.scheduler(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer,
        #     max_lr=self.hparams.base_lr,
        #     # automatically calculate total setps in pytorch-lightning trainer
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=self.hparams.warmup_epochs / self.hparams.epochs,
        #     anneal_strategy="linear",
        #     div_factor=self.hparams.base_lr / self.hparams.warmup_lr,
        #     final_div_factor=1e6,
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # unit to change learning rate(step / epoch)
                "frequency": 1,  # frequency for changing learning rate
            },
        }

    def optimizer_zero_grad(
        self,
        _epoch: int,
        _batch_idx: int,
        optimizer: torch.optim.Optimizer,
        _optimizer_idx: int,
    ):
        optimizer.zero_grad(set_to_none=True)
