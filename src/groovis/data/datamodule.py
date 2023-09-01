from typing import Optional, Type

from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader

from src.groovis.data.augmentation import SIMCLR_AUG_RELAXED
from src.groovis.data.dataset import BaseImagenet
from src.groovis.schemas import Cfg


class ImagenetModule(LightningDataModule):
    hparams: Cfg
    trainer: Optional[Trainer]

    def __init__(self, config: Cfg, dataset: Type[BaseImagenet]):
        super().__init__()
        self.save_hyperparameters(config)
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None):
        self.dataset(split="train")
        self.dataset(split="validation")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(transforms=SIMCLR_AUG_RELAXED, split="train"),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            # control number of processor for parallel processing
            num_workers=0 if self.on_cpu else self.trainer.devices * 4,
            # number of batch to pull before next training step
            prefetch_factor=2,
            # aggregate batch tensor to pass that to GPU right away
            pin_memory=not self.on_cpu,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(transforms=SIMCLR_AUG_RELAXED, split="validation"),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0 if self.on_cpu else self.trainer.devices * 4,
            prefetch_factor=2,
            pin_memory=not self.on_cpu,
        )

    @property
    def on_cpu(self) -> bool:
        return self.trainer.accelerator == "cpu"
