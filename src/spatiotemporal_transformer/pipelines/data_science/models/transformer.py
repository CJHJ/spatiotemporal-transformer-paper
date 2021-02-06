"""Transformer"""
import copy
from typing import List

import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange


class LinearProjection(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class PositionEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        period: int = 10000,
        max_seq_len: int = 80,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.period = period
        self.use_cuda = use_cuda

        embed = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(d_model):
                embed[pos, i] = self.encoder(i, pos)
        embed = embed.unsqueeze(0)  # batch size
        self.register_buffer("embed", embed)

    def encoder(self, i: int, pos: int):
        wt = pos / (self.period ** (2 * i / self.d_model))
        if i % 2 == 0:
            return np.sin(wt)
        return np.cos(wt)

    def forward(self, x: torch.Tensor):
        x = x * np.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.embed[:, :seq_len], requires_grad=False)
        if self.use_cuda:
            x = x.cuda()

        return x


class PositionEmbedding(nn.Module):
    def __init__(
        self, d_model: int, time_length: int, use_cuda: bool = False,
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.embed = nn.Parameter(torch.randn(1, time_length, d_model))

    def forward(self, x: torch.Tensor):
        return x + self.embed


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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = F.gelu(self.linear_2(x))

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout: float = 0):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.h = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        d_k: int,
        mask=None,
        dropout=None,
    ):
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        return scores @ V

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None
    ):
        bs = Q.size(0)

        K = self.k_linear(K).view(bs, -1, self.h, self.d_k)
        Q = self.q_linear(Q).view(bs, -1, self.h, self.d_k)
        V = self.v_linear(V).view(bs, -1, self.h, self.d_k)

        K = K.transpose(1, 2)
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = self.attention(Q, K, V, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        return self.out(concat)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiheadAttention(num_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_model * 2, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiheadAttention(num_heads, d_model, dropout=dropout)
        self.attn_2 = MultiheadAttention(num_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_model * 2, dropout=dropout)
        if use_cuda:
            self.ff = self.ff.cuda()

    def forward(
        self, x: torch.Tensor, encoder_outputs: int, source_mask, target_mask
    ):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, target_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(
            self.attn_2(x2, encoder_outputs, encoder_outputs, source_mask)
        )
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))

        return x


def get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        time_length: int,
        d_model: int,
        N: int,
        num_heads: int,
        use_cuda: bool,
        dropout: float = 0.0,
        embed_mode: str = "Embedding",
    ):
        super().__init__()
        self.N = N
        self.embed = LinearProjection(input_size, d_model)
        if embed_mode == "Embedding":
            self.position_embed = PositionEmbedding(
                d_model, time_length, use_cuda=use_cuda
            )
        elif embed_mode == "Sin":
            self.position_embed = PositionEncoding(d_model, use_cuda=use_cuda)
        self.layers = get_clones(
            EncoderLayer(d_model, num_heads, dropout=dropout), N
        )
        self.norm = Norm(d_model)

    def forward(self, source: torch.Tensor, mask: torch.Tensor):
        x = self.embed(source)
        x = self.position_embed(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        time_length: int,
        d_model: int,
        N: int,
        num_heads: int,
        use_cuda: bool,
        dropout: float = 0.0,
        embed_mode: str = "Embedding",
    ):
        super().__init__()
        self.N = N
        self.embed = LinearProjection(input_size, d_model)
        if embed_mode == "Embedding":
            self.position_embed = PositionEmbedding(
                d_model, time_length, use_cuda=use_cuda
            )
        elif embed_mode == "Sin":
            self.position_embed = PositionEncoding(d_model, use_cuda=use_cuda)
        self.layers = get_clones(
            DecoderLayer(
                d_model, num_heads, use_cuda=use_cuda, dropout=dropout
            ),
            N,
        )
        self.norm = Norm(d_model)

    def forward(self, target, encoder_outputs, source_mask, target_mask):
        x = self.embed(target)
        x = self.position_embed(x)
        for i in range(self.N):
            x = self.layers[i](x, encoder_outputs, source_mask, target_mask)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        input_size,
        target_size,
        time_length,
        d_model,
        N,
        num_heads: int,
        use_cuda: bool = False,
        dropout: float = 0.2,
        embed_mode: str = "Embedding",  # "Embedding" or "Sin"
        quantiles: List[float] = [],
    ):
        super().__init__()
        self.encoder = Encoder(
            input_size,
            time_length,
            d_model,
            N,
            num_heads,
            use_cuda,
            dropout=dropout,
            embed_mode=embed_mode,
        )
        self.decoder = Decoder(
            target_size,
            time_length,
            d_model,
            N,
            num_heads,
            use_cuda,
            dropout=dropout,
            embed_mode=embed_mode,
        )
        self.n_quantiles = len(quantiles)
        if quantiles:
            self.out = nn.Linear(d_model, target_size * self.n_quantiles)
        else:
            self.out = nn.Linear(d_model, target_size)

    def forward(self, input_data, target_data, input_mask, target_mask):
        encoder_outputs = self.encoder(input_data, input_mask)
        decoder_output = self.decoder(
            target_data, encoder_outputs, input_mask, target_mask
        )
        output = F.sigmoid(self.out(decoder_output))

        if self.n_quantiles > 0:
            output = rearrange(
                output, "b t (q d) -> b t q d", q=self.n_quantiles
            )

        return output


if __name__ == "__main__":
    batch_size = 10
    feature_size = 10
    max_seq_len = 80
    target_seq_len = 80
    d_model = 512
    pos_period = 10000
    num_heads = 2
    num_layers = 6
    test_tensor = torch.zeros((batch_size, max_seq_len, feature_size))
    target_tensor = torch.zeros((batch_size, target_seq_len, feature_size))

    # Initialization
    linear = LinearProjection(feature_size, d_model)
    pe = PositionEncoding(d_model, period=pos_period, max_seq_len=max_seq_len)
    encoder_layer = EncoderLayer(d_model, num_heads=num_heads)
    decoder_layer = DecoderLayer(d_model, num_heads=num_heads)

    # Linear embedding
    x = linear(test_tensor)
    target_x = linear(target_tensor)

    # Position embedding
    x = pe(x)
    target_x = pe(target_x)

    # Encoder
    mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]))
    x = encoder_layer(x, mask)

    # Decoder
    target_mask = np.triu(
        np.ones((1, target_seq_len, target_seq_len)), k=1
    ).astype(np.int8)
    target_mask = Variable(torch.from_numpy(target_mask) == 0)
    x = decoder_layer(target_x, x, mask, target_mask)

    # Transformer
    transformer = Transformer(
        feature_size, feature_size, d_model, num_layers, num_heads
    )
    transformer(test_tensor, target_tensor, mask, target_mask)

