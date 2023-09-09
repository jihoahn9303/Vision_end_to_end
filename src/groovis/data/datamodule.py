from typing import Optional

from hydra_zen.typing import Partial
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from torch.utils.data import DataLoader

from src.groovis.data.dataset import BaseImagenet


class ImagenetModule(LightningDataModule):
    trainer: Trainer

    def __init__(
        self, dataloader: Partial[DataLoader], dataset: Partial[BaseImagenet]
    ) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None):
        self.dataset(split="train")
        self.dataset(split="validation")

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            dataset=self.dataset(split="train"),
            **(
                {
                    # control number of processor for parallel processing
                    "num_workers": self.trainer.num_devices * 4,
                    # number of batch to pull before next training step
                    "prefetch_factor": 2,
                    "persistent_workers": True,
                    # aggregate batch tensor to pass that to GPU right away
                    "pin_memory": True,
                }
                if not self.on_cpu
                else {}
            )
        )

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(
            dataset=self.dataset(split="validation"),
            **(
                {
                    "num_workers": self.trainer.num_devices * 4,
                    "prefetch_factor": 2,
                    "persistent_workers": True,
                    "pin_memory": True,
                }
                if not self.on_cpu
                else {}
            )
        )

    @property
    def on_cpu(self) -> bool:
        return isinstance(self.trainer.accelerator, CPUAccelerator)
