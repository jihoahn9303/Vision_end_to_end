from einops.layers.torch import EinMix
from hydra_zen.typing import Partial
from torch import nn

from src.groovis.models.components.layer_norm import NormType
from src.groovis.types import (
    SequenceTensor,
    SequenceToSequence,
    StrictFloat,
    StrictInt,
    torchtyped,
)


class MixerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: StrictInt = 1024,
        expansion_factor: StrictFloat = 4,
        act_layer: Partial[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.block: SequenceToSequence = nn.Sequential(
            EinMix(
                "b n d_in -> b n d_out",
                weight_shape="d_in d_out",
                bias_shape="d_out",
                d_in=embed_dim,
                d_out=int(expansion_factor * embed_dim),
            ),
            act_layer(),
            EinMix(
                "b n d_out -> b n d_in",
                weight_shape="d_out d_in",
                bias_shape="d_in",
                d_in=int(expansion_factor * embed_dim),
                d_out=embed_dim,
            ),
        )

        self.norm = nn.LayerNorm(embed_dim)

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        # residual connection
        # example) keep background and dive deeply into
        # some region that we really want to explore
        return representation + self.block(self.block(representation))  # Pre-norm


class Mixer(nn.Module):
    def __init__(
        self, block: Partial[nn.Module], norm: Partial[NormType], depth: StrictInt = 24
    ) -> None:
        super().__init__()

        # wrap list of nn.Module by nn.ModuleList
        # for working __setattr__ in nn.Module
        self.blocks = nn.ModuleList([norm(block=block()) for _ in range(depth)])

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        for block in self.blocks:
            representation = block(representation)

        return representation


# mixer = Mixer(depth=24)
# print(sum(parameter.numel() for parameter in mixer.parameters()))
