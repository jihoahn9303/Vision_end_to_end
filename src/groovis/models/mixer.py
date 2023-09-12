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


class PerTokenMixerBlock(nn.Module):
    """
    Block for per-location mixing operations
    """

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
                d_in=embed_dim,
                d_out=int(expansion_factor * embed_dim),
            ),
        )

        self.norm = nn.LayerNorm(embed_dim)

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        return self.block(representation)


class CrossTokenMixerBlock(nn.Module):
    """
    Block for cross-location mixing operations
    """

    def __init__(
        self,
        seq_len: StrictInt = 196,
        expansion_factor: StrictFloat = 0.5,
        act_layer: Partial[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.block: SequenceToSequence = nn.Sequential(
            EinMix(
                "b n_in d -> b n_out d",
                weight_shape="n_in n_out",
                bias_shape="n_out",
                d_in=seq_len,
                d_out=int(expansion_factor * seq_len),
            ),
            act_layer(),
            EinMix(
                "b n_out d -> b n_in d",
                weight_shape="n_out n_in",
                bias_shape="n_in",
                d_in=seq_len,
                d_out=int(expansion_factor * seq_len),
            ),
        )

        self.norm = nn.LayerNorm(seq_len)

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        return self.block(representation)


class Mixer(nn.Module):
    def __init__(
        self,
        per_location_block: Partial[nn.Module],
        cross_location_block: Partial[nn.Module],
        norm: Partial[NormType],
        depth: StrictInt = 24,
    ) -> None:
        super().__init__()

        # wrap list of nn.Module by nn.ModuleList
        # for working __setattr__ in nn.Module
        # self.blocks = nn.ModuleList([norm(block=block()) for _ in range(depth)])
        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(norm(block=per_location_block()))
            self.blocks.append(norm(block=cross_location_block()))

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        for block in self.blocks:
            representation = block(representation)

        return representation


# mixer = Mixer(depth=24)
# print(sum(parameter.numel() for parameter in mixer.parameters()))
