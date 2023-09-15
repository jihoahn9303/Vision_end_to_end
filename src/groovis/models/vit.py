import torch
from einops import einsum, pack, rearrange, unpack
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


class MLPBlock(nn.Module):
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


class SelfAttention(nn.Module):
    """
    Self-Attention block
    """

    def __init__(self, embed_dim: StrictInt = 1024, num_heads: StrictInt = 8) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.to_qkv = EinMix(
            "batch sequence dim_in -> batch sequence dim_out",
            weight_shape="dim_in dim_out",
            bias_shape="dim_out",
            dim_in=embed_dim,
            dim_out=3 * self.embed_dim,
        )

        self.projection = EinMix(
            "batch sequence dim_out -> batch sequence dim_in",
            weight_shape="dim_out dim_in",
            bias_shape="dim_in",
            dim_in=self.embed_dim,
            dim_out=self.embed_dim,
        )

    def forward(self, representation: SequenceTensor):
        query, key, value = unpack(
            tensor=self.to_qkv(representation),
            packed_shapes=[[self.embed_dim], [self.embed_dim], [self.embed_dim]],
            pattern="b s *",
        )

        query, key, value = map(
            lambda x: rearrange(x, "b s (h d) -> b h s d", h=self.num_heads),
            (query, key, value),
        )

        score = (
            einsum(query, key, "b h q d, b h k d -> b h q k") * self.head_dim**-0.5
        )

        attention_score = score.softmax(dim=-1)

        final = einsum(attention_score, value, "b h q k, b h k d-> b h q d")
        final = rearrange(final, "b h q d -> b q (h d)")

        # Fully connected layer for residual connection
        final = self.projection(final)

        return final


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
                n_in=seq_len,
                n_out=int(expansion_factor * seq_len),
            ),
            act_layer(),
            EinMix(
                "b n_out d -> b n_in d",
                weight_shape="n_out n_in",
                bias_shape="n_in",
                n_in=seq_len,
                n_out=int(expansion_factor * seq_len),
            ),
        )

        self.norm = nn.LayerNorm(seq_len)

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        return self.block(representation)


# Implementation for paper named
# Scaling Vision Transformers to 22 Billion Parameters
class FusedTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: StrictInt = 1024,
        expansion_factor: StrictFloat = 4,
        num_heads: StrictInt = 8,
        act_layer: Partial[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.expanded_dim = int(expansion_factor * embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = num_heads * self.head_dim

        self.projection_in = EinMix(
            pattern="batch sequence d_in -> batch sequence d_out",
            weight_shape="d_in d_out",
            d_in=embed_dim,
            d_out=self.expanded_dim + 3 * embed_dim,
        )
        self.mlp_bias = nn.Parameter(torch.zeros(self.expanded_dim))

        self.projection_out = EinMix(
            pattern="batch sequence d_out -> batch sequence d_in",
            weight_shape="d_out d_in",
            bias_shape="d_in",
            d_in=embed_dim,
            d_out=self.expanded_dim + embed_dim,
        )

        self.act_layer = act_layer()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        # get first MLP Layer output and query/key/value together
        mlp_1st_tensor, query, key, value = unpack(
            tensor=self.projection_in(representation),
            packed_shapes=[
                [self.expanded_dim],
                [self.embed_dim],
                [self.embed_dim],
                [self.embed_dim],
            ],
            pattern="b s *",
        )

        # get 2nd MLP input
        mlp_2nd_tensor_input = self.act_layer(mlp_1st_tensor + self.mlp_bias)

        # get attention output
        query, key, value = map(
            lambda x: rearrange(x, "b s (h d) -> b h s d", h=self.num_heads),
            (self.query_norm(query), self.key_norm(key), value),
        )
        score = einsum(query, key, "b h q d, b h k d -> b h q k") * (
            self.head_dim**-0.5
        )
        attention_score = score.softmax(dim=-1)
        attention_output = einsum(attention_score, value, "b h q k, b h k d -> b h q d")
        attention_output = rearrange(attention_output, "b h q d -> b q (h d)")

        # fusion(get 2nd MLP output + final attention output)
        output, _packed_shape = pack(
            [mlp_2nd_tensor_input, attention_output], pattern="b s *"
        )
        output = self.projection_out(output)

        return output


# Implementation for MLP-Mixer and Self-attention
class AlternatingBackbone(nn.Module):
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
            self.blocks.append(norm(block=cross_location_block()))
            self.blocks.append(norm(block=per_location_block()))

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        for block in self.blocks:
            representation = block(representation)

        return representation


class HomogeneousBackbone(nn.Module):
    def __init__(
        self,
        block: Partial[nn.Module],
        norm: Partial[NormType],
        depth: StrictInt = 24,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([norm(block=block()) for _ in range(depth)])

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        for block in self.blocks:
            representation = block(representation)

        return representation
