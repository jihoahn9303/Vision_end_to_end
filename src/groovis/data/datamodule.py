from typing import Optional

from hydra_zen.typing import Partial
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from torch.utils.data import DataLoader

from src.groovis.data.dataset import BaseImagenet


class ImagenetModule(LightningDataModule):
    trainer: Optional[Trainer]

    def __init__(self, dataloader: Partial[DataLoader], dataset: Partial[BaseImagenet]):
        super().__init__()
        self.dataloader = dataloader
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None):
        self.dataset(split="train")
        self.dataset(split="validation")

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            dataset=self.dataset(split="train"),
            # control number of processor for parallel processing
            num_workers=None if self.on_cpu else self.trainer.num_devices * 4,
            # number of batch to pull before next training step
            prefetch_factor=2,
            # aggregate batch tensor to pass that to GPU right away
            pin_memory=None if self.on_cpu else True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(split="validation"),
            num_workers=None if self.on_cpu else self.trainer.num_devices * 4,
            prefetch_factor=2,
            pin_memory=None if self.on_cpu else True,
        )

    @property
    def on_cpu(self) -> bool:
        return isinstance(self.trainer.accelerator, CPUAccelerator)
