import torch
from einops.layers.torch import EinMix
from torch import nn

from src.groovis.types import (
    ImageTensor,
    ImageToSequence,
    SequenceTensor,
    StrictInt,
    torchtyped,
)

# from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: StrictInt = 224,
        patch_size: StrictInt = 16,
        channels: StrictInt = 3,
        embed_dim: StrictInt = 1024,
    ) -> None:
        super().__init__()

        # module for rearrange
        # self.split_images = Rearrange(
        #     "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size
        # )

        # module for weight averaging
        # self.projection = nn.Linear(
        #     in_features=channels * patch_size**2, out_features=embed_dim, bias=True
        # )

        # merge two step codes into one step code using EinMix
        """
        b: batch size
        c: number of channels
        h: h'th patch(height)
        w: w'th patch(width)
        ph: ph'th pixel in 'h'th height  + 'w'th width patch
        pw: pw'th pixel in 'h'th height  + 'w'th width patch
        d: embedding size
        """
        self.patch_embed: ImageToSequence = EinMix(
            pattern="b c (h ph) (w pw) -> b (h w) d",
            weight_shape="c ph pw d",
            bias_shape="d",
            c=channels,
            ph=patch_size,
            pw=patch_size,
            d=embed_dim,
        )

        self.position_embedding = nn.Parameter(
            torch.randn((image_size // patch_size) ** 2, embed_dim) * 0.01
        )

    @torchtyped
    def forward(self, images: ImageTensor) -> SequenceTensor:
        return self.patch_embed(images) + self.position_embedding

        # Below code is equivalent to self.projection(patches)
        # representation = torch.einsum(
        #     "b n p, d p -> b n d",
        #     patches,
        #     self.weight
        # )

        # representation += repeat(
        #     self.bias,
        #     "d -> b n d",
        #     b=representation.shape[0],
        #     n=representation.shape[1]
        # )

        # return representation
