from typing import Optional, Type

from dataset import BaseImagenet
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.groovis.data.augmentation import SIMCLR_AUG_RELAXED
from src.groovis.schemas import Cfg


class ImagenetModule(LightningDataModule):
    hparams: Cfg

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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(transforms=SIMCLR_AUG_RELAXED, split="validation"),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
        )
