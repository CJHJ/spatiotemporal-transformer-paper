"""Spatiotemporal Transformer"""
# %%
import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


def divide_to_patches(x, patch_size):
    return rearrange(
        x,
        "b t (h p1) (w p2) -> b t (h w) (p1 p2)",
        p1=patch_size,
        p2=patch_size,
    )


def merge_from_patches(x, num_patch_width, patch_size, n_quantiles: int = 1):
    return torch.squeeze(
        rearrange(
            x,
            "b t (h w) (q p1 p2) -> b t q (h p1) (w p2)",
            q=n_quantiles,
            h=num_patch_width,
            p1=patch_size,
            p2=patch_size,
        ),
        dim=2,
    )


def get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )

        return norm


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        time_length: int,
        num_patch: int,
        num_heads: int,
        d_model: int,
        device: str = "cpu",
        dropout: float = 0.0,
        temporal_pos_embed: nn.Parameter = None,
        decoder: bool = False,
    ):
        super().__init__()
        self.device = device
        self.time_length = time_length
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // self.num_heads
        self.decoder = decoder

        # Models
        self.input_norm = Norm(d_model)

        # Attentions - Encoder
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.attention_norm = Norm(d_model)
        self.dropout_attn = nn.Dropout(dropout)

        # Output layers - TODO: currently average of linear layers, 1D conv?, another attention layer?
        self.temporal_pos_embed = temporal_pos_embed
        self.temporal_concat_linear = nn.ModuleList()
        for i in range(self.time_length):
            self.temporal_concat_linear.append(nn.Linear(d_model, d_model))
        self.dropout_temporal_concat = nn.Dropout(dropout)
        self.output_norm = Norm(d_model)

        if self.decoder:
            # Attentions - Decoder
            self.linear_q_dec = nn.Linear(d_model, d_model)
            self.linear_k_dec = nn.Linear(d_model, d_model)
            self.linear_v_dec = nn.Linear(d_model, d_model)
            self.attention_norm_dec = Norm(d_model)
            self.dropout_attn_dec = nn.Dropout(dropout)

            self.temporal_concat_linear_dec = nn.ModuleList()
            for i in range(self.time_length):
                self.temporal_concat_linear_dec.append(
                    nn.Linear(d_model, d_model)
                )
            self.dropout_temporal_concat_dec = nn.Dropout(dropout)
            self.output_norm_dec = Norm(d_model)

        # Feed forward layer
        self.ff_1 = nn.Linear(d_model, d_model * 2)
        self.dropout_mid = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(d_model * 2, d_model)
        self.dropout_ff = nn.Dropout(dropout)

    def calc_attention(
        self,
        bs,
        Q,
        K,
        V,
        mask,
        linear_q,
        linear_k,
        linear_v,
        dropout_attn,
        attention_norm,
        temporal_concat_linear,
    ):
        # Attention weights
        Q = linear_q(Q).view(
            bs, self.time_length, -1, self.num_heads, self.d_head
        )
        K = linear_k(K).view(
            bs, self.time_length, -1, self.num_heads, self.d_head
        )
        V = linear_v(V).view(
            bs, self.time_length, -1, self.num_heads, self.d_head
        )
        Q = Q.transpose(2, 3)
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)
        QK_temporal = torch.einsum(
            "b t i j k, b u i l k -> b t u i j l", Q, K
        ) / np.sqrt(self.d_head)
        if mask is not None:
            QK_temporal = QK_temporal.masked_fill(mask == 0, -1e9)
        QK_temporal = dropout_attn(F.softmax(QK_temporal, dim=-1,))
        if (
            mask is not None
        ):  # Prevent information leak by 0-ing the attention scores
            QK_temporal = QK_temporal.masked_fill(mask == 0, 0)
        AV_temporal = torch.einsum(
            "b t u i j l, b u i l k -> b t u i j k", QK_temporal, V
        )

        # Head concatenation
        AV_concat = (
            AV_temporal.transpose(3, 4)
            .contiguous()
            .view(bs, self.time_length, self.time_length, -1, self.d_model)
        )
        AV_concat = attention_norm(AV_concat)

        # Temporal embedding
        x_2 = AV_concat + self.temporal_pos_embed[:, :, :]

        # Temporal concat
        temporal_concat = torch.zeros(
            bs, self.time_length, self.num_patch, self.d_model
        ).to(self.device)
        for i in range(self.time_length):
            temporal_concat += temporal_concat_linear[i](
                x_2[:, :, i, :, :] / self.time_length
            )

        return F.gelu(temporal_concat)

    def forward(
        self, x, encoder_outputs=None, input_mask=None, target_mask=None
    ):
        bs = x.shape[0]

        x_2 = self.input_norm(x)
        if self.decoder:
            first_mask = target_mask
        else:
            first_mask = input_mask
        x = x + self.dropout_temporal_concat(
            self.calc_attention(
                bs,
                x_2,
                x_2,
                x_2,
                first_mask,
                self.linear_q,
                self.linear_k,
                self.linear_v,
                self.dropout_attn,
                self.attention_norm,
                self.temporal_concat_linear,
            )
        )
        x_2 = self.output_norm(x)
        if self.decoder:
            x = x + self.dropout_temporal_concat_dec(
                self.calc_attention(
                    bs,
                    x_2,
                    encoder_outputs,
                    encoder_outputs,
                    input_mask,
                    self.linear_q_dec,
                    self.linear_k_dec,
                    self.linear_v_dec,
                    self.dropout_attn_dec,
                    self.attention_norm_dec,
                    self.temporal_concat_linear_dec,
                )
            )
            x_2 = self.output_norm_dec(x)

        # Feed forward
        x_2 = self.dropout_mid(F.gelu(self.ff_1(x_2)))
        x_2 = F.gelu(self.ff_2(x_2))
        x = x + self.dropout_ff(x_2)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        time_length: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        patch_feature_size: int,
        num_patch: int,
        device: str = "cpu",
        dropout: float = 0.0,
        spatial_pos_embed: nn.Parameter = None,
        temporal_pos_embed: nn.Parameter = None,
    ):
        super().__init__()
        self.device = device
        self.time_length = time_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_feature_size = patch_feature_size
        self.num_patch = num_patch
        self.dropout = dropout

        # Patching layer
        self.space_linear_proj = nn.Linear(self.patch_feature_size, d_model)
        self.spatial_pos_embed = spatial_pos_embed
        self.temporal_pos_embed = temporal_pos_embed

        # Attention layers
        self.attn_layers = get_clones(
            MultiHeadAttention(
                self.time_length,
                self.num_patch,
                num_heads,
                self.d_model,
                self.device,
                self.dropout,
                self.temporal_pos_embed,
            ),
            self.num_layers,
        )

        # Unpatch layer
        self.norm = Norm(self.d_model)

    def forward(self, x, input_mask):
        bs = x.shape[0]

        # Patch division and projection
        x = divide_to_patches(x, self.patch_size)
        x = self.space_linear_proj(
            x.reshape([bs, self.time_length, self.num_patch, -1])
        )
        x += self.spatial_pos_embed[:, :, :]

        # Spatiotemporal Attention
        for i in range(self.num_layers):
            x = self.attn_layers[i](x)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        time_length: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        patch_feature_size: int,
        num_patch: int,
        device: str = "cpu",
        dropout: float = 0.0,
        spatial_pos_embed: nn.Parameter = None,
        temporal_pos_embed: nn.Parameter = None,
        quantiles: List[float] = None,
    ):
        super().__init__()
        self.device = device
        self.time_length = time_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_feature_size = patch_feature_size
        self.num_patch = num_patch
        self.dropout = dropout
        self.quantiles = quantiles

        # Patching layer
        self.space_linear_proj = nn.Linear(patch_feature_size, d_model)
        self.spatial_pos_embed = spatial_pos_embed
        self.temporal_pos_embed = temporal_pos_embed

        # Attention layers
        self.attn_layers = get_clones(
            MultiHeadAttention(
                self.time_length,
                self.num_patch,
                num_heads,
                self.d_model,
                self.device,
                self.dropout,
                self.temporal_pos_embed,
                decoder=True,
            ),
            self.num_layers,
        )

        # Unpatch layer
        self.norm = Norm(self.d_model)
        if self.quantiles:
            self.unpatch_layer = nn.Linear(
                d_model, patch_feature_size * len(self.quantiles)
            )
        else:
            self.unpatch_layer = nn.Linear(d_model, patch_feature_size)

    def forward(self, x, encoder_outputs, input_mask, target_mask):
        bs, t, h, w = x.shape

        # Patch division and projection
        x = divide_to_patches(x, self.patch_size)
        x = self.space_linear_proj(
            x.reshape([bs, self.time_length, self.num_patch, -1])
        )
        x += self.spatial_pos_embed[:, :, :]

        # Spatiotemporal Attention
        for i in range(self.num_layers):
            x = self.attn_layers[i](
                x, encoder_outputs, input_mask, target_mask
            )

        # Unpatch layer
        x = self.unpatch_layer(self.norm(x))
        num_patch_width = int(np.sqrt(self.num_patch))

        return merge_from_patches(
            x,
            num_patch_width,
            self.patch_size,
            len(self.quantiles) if self.quantiles else 1,
        )


class SpatiotemporalTransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        input_width: int,
        input_height: int,
        time_length: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        use_cuda: bool = False,
        dropout: float = 0.0,
        quantiles: List[float] = None,
    ):
        super().__init__()
        self.width = input_width
        self.height = input_height
        input_size = self.width * self.height

        self.time_length = time_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_feature_size = patch_size ** 2
        self.num_patch = input_size // self.patch_feature_size

        self.device = "cuda" if use_cuda else "cpu"
        self.dropout = dropout
        self.quantiles = quantiles

        # Patch positional embedder
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_patch, d_model)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.time_length, self.num_patch, d_model)
        )

        # Encoder
        self.encoder = Encoder(
            time_length=self.time_length,
            patch_size=self.patch_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            patch_feature_size=self.patch_feature_size,
            num_patch=self.num_patch,
            device=self.device,
            dropout=self.dropout,
            spatial_pos_embed=self.spatial_pos_embed,
            temporal_pos_embed=self.temporal_pos_embed,
        )

        # Decoder
        self.decoder = Decoder(
            time_length=self.time_length,
            patch_size=self.patch_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            patch_feature_size=self.patch_feature_size,
            num_patch=self.num_patch,
            device=self.device,
            dropout=self.dropout,
            spatial_pos_embed=self.spatial_pos_embed,
            temporal_pos_embed=self.temporal_pos_embed,
            quantiles=self.quantiles,
        )

    def forward(self, input_data, target_data, input_mask, target_mask):
        encoder_outputs = self.encoder(input_data, input_mask)
        decoder_outputs = self.decoder(
            target_data, encoder_outputs, input_mask, target_mask
        )

        return decoder_outputs


if __name__ == "__main__":
    # For model viz
    writer = SummaryWriter("tensorboard_runs/spatiotemporal_seq2seq")

    # Parameters
    bs, t, w, h = 1, 5, 5, 5
    patch_size = 1
    d_model = 8
    num_heads = 8
    num_patch = (w // patch_size) * (h // patch_size)
    device = "cpu"

    # Data
    input_data = torch.ones(bs, t, h, w)
    for i in range(5):
        input_data[:, i, :, :] = input_data[:, i, :, :] * i
    target_data = torch.ones(bs, t, h, w)
    for i in range(4, 9):
        target_data[:, i - 4, :, :] = target_data[:, i - 4, :, :] * i
    gt_data = torch.ones(bs, t, h, w)
    for i in range(5, 10):
        gt_data[:, i - 5, :, :] = gt_data[:, i - 5, :, :] * i
    input_mask = torch.ones(bs, t, t, num_heads, num_patch, num_patch)

    # MASKING - create a 2D time length mask and then do a product with torch.ones
    target_mask = np.triu(np.ones((1, t, t)), k=1).astype(
        np.int8
    )  # 0 to 1, 1 to 0 (transpose)
    target_mask = torch.from_numpy(target_mask) == 0
    target_mask = torch.einsum(
        "b t u, n w h -> b t u n w h",
        target_mask,
        torch.ones((num_heads, num_patch, num_patch)).to(device),
    )

    # Init model
    attention = SpatiotemporalTransformerSeq2Seq(
        w, h, t, patch_size, d_model, num_heads, 3, False, 0.0
    )
    attention = attention.to(device)

    # For model viz
    # writer.add_graph(
    #     attention,
    #     (
    #         input_data.to(device),
    #         target_data.to(device),
    #         input_mask,
    #         target_mask,
    #     ),
    # )

    # test = attention(
    #     input_data.to(device), target_data.to(device), input_mask, target_mask,
    # )

    # Test model
    for i in tqdm(range(3)):
        print(f"EPOCH {i}")
        test = attention(
            input_data.to(device),
            target_data.to(device),
            input_mask,
            target_mask,
        )
        loss = F.mse_loss(test, gt_data.to(device))
        loss.backward()

