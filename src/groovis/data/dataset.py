import os
from pathlib import Path
from typing import Literal, Union

import albumentations as A
import datasets as D
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from src.groovis.utils import image_path_to_array

Splits = Literal["train", "validation"]
IMG_EXTENSIONS = [".webp", ".jpg", ".jpeg", ".png"]


class Animals(Dataset):
    def __init__(self, root: str, transforms: A.Compose):
        self.paths = [
            path for path in Path(root).iterdir() if path.suffix in IMG_EXTENSIONS
        ]
        self.transforms = transforms

    def __getitem__(self, index) -> list[torch.Tensor]:
        # return image_path_to_tensor(self.paths[index])
        image = image_path_to_array(self.paths[index])

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return len(self.paths)


class BaseImagenet(Dataset):
    dataset: Union[D.DatasetDict, D.Dataset, D.IterableDatasetDict, D.IterableDataset]

    def __init__(self, split: Splits, transforms: A.Compose):
        self.transforms = transforms
        self.set_dataset(split=split)

    def __getitem__(self, index) -> list[torch.Tensor]:
        # 라벨 정보를 제외하고 이미지 정보만 가져옴.
        image: Image.Image = self.dataset[index]["image"]
        # 흑백 이미지도 존재할 수 있으므로, batch 단위 전처리를 위하여 채널을 통일함.
        image = image.convert("RGB")
        image = np.array(image)

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return self.dataset.num_rows

    def set_dataset(self, split: Splits):
        raise NotImplementedError


class Imagenette(BaseImagenet):
    def __init__(self, split: Splits, transforms: A.Compose):
        super().__init__(transforms, split)

    def set_dataset(self, split: Splits):
        self.dataset = load_dataset(path="frgfm/imagenette", name="320px", split=split)


class Imagenet(BaseImagenet):
    def __init__(self, split: Splits, transforms: A.Compose):
        super().__init__(transforms, split)

    def set_dataset(self, split: Splits):
        if "HF_AUTH_TOKEN" not in os.environ:
            raise KeyError("'HF_AUTH_TOKEN' must be set.")

        self.dataset = load_dataset(
            path="imagenet-1k", split=split, use_auth_token=os.environ["HF_AUTH_TOKEN"]
        )
