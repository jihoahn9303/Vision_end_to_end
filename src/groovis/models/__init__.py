from typing import Optional

import torch
from einops.layers.torch import Reduce

# from einops.layers.torch import Rearrange
from torch import nn

from src.groovis.types import (
    ImageTensor,
    ImageToSequence,
    PooledTensor,
    SequenceToPooled,
    SequenceToSequence,
)


class Architecture(nn.Module):
    def __init__(self, patch_embed: nn.Module, backbone: Optional[nn.Module]) -> None:
        super().__init__()

        # module for patch embedding
        self.patch_embed: ImageToSequence = patch_embed

        # module for backbone network
        self.backbone: SequenceToSequence = backbone or nn.Identity()

        # module for pooling
        self.pool: SequenceToPooled = Reduce("b n d -> b d", "mean")

    def forward(self, images: ImageTensor) -> PooledTensor:
        representation = self.patch_embed(images)
        # representation = reduce(representation, "b n d-> b d", "mean")   # pooling
        representation = self.backbone(representation)
        representation = self.pool(representation)

        return representation
