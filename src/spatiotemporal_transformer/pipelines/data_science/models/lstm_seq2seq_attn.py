"""LSTM Seq2Seq Attention"""
from typing import List
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM

from einops import rearrange


class Encoder(nn.Module):
    def __init__(
        self,
        h_0: nn.Parameter,
        c_0: nn.Parameter,
        linear_output_size: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = LSTM(
            linear_output_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.h_0 = h_0
        self.c_0 = c_0

    def forward(self, x):
        batch_size = x.shape[0]

        h_n = self.h_0.expand(-1, batch_size, -1).contiguous()
        c_n = self.c_0.expand(-1, batch_size, -1).contiguous()
        output, (hidden_output, cell_output) = self.lstm(x, (h_n, c_n))

        return output, hidden_output, cell_output


class Decoder(nn.Module):
    def __init__(
        self,
        linear_output_size: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = LSTM(
            linear_output_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x, output_encoder, hidden_encoder, cell_encoder):
        lstm_output, _ = self.lstm(x, (hidden_encoder, cell_encoder))
        attention_score = torch.softmax(
            torch.einsum("btd, bud -> btu", lstm_output, output_encoder)
            / np.sqrt(lstm_output.shape[-1]),
            dim=-1,
        )  # QK^T
        final_attention = torch.einsum(
            "btu, bud -> btd", attention_score, output_encoder
        )
        lstm_output = torch.cat((lstm_output, final_attention), dim=-1)

        return lstm_output


class LSTMSeq2SeqAttn(nn.Module):
    def __init__(
        self,
        feature_size: int,
        linear_output_size: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        one_step_ahead: bool,
        use_cuda: bool,
        dropout: float = 0.2,
        sigmoid_output: bool = False,
        quantiles: List[float] = [],
    ):
        super().__init__()
        self.one_step_ahead = one_step_ahead
        self.sigmoid_output = sigmoid_output

        self.linear_in = nn.Linear(feature_size, linear_output_size)
        self.h_0 = nn.Parameter(
            torch.zeros(lstm_num_layers, 1, lstm_hidden_size)
        )
        self.c_0 = nn.Parameter(
            torch.zeros(lstm_num_layers, 1, lstm_hidden_size)
        )
        self.encoder = Encoder(
            h_0=self.h_0,
            c_0=self.c_0,
            linear_output_size=linear_output_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            linear_output_size=linear_output_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
        )
        self.n_quantiles = len(quantiles)

        if quantiles:
            self.linear_out = nn.Linear(
                lstm_hidden_size * 2, feature_size * self.n_quantiles
            )
        else:
            self.linear_out = nn.Linear(lstm_hidden_size * 2, feature_size)

    def one_step_ahead_forward(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ):
        input_batch = F.gelu(self.linear_in(input_batch))
        output_encoder, hidden_encoder, cell_encoder = self.encoder(
            input_batch
        )

        target_batch = F.gelu(self.linear_in(target_batch))
        decoder_output = self.decoder(
            target_batch, output_encoder, hidden_encoder, cell_encoder
        )

        output = self.linear_out(decoder_output)

        if self.n_quantiles > 0:
            output = rearrange(
                output, "b t (q d) -> b t q d", q=self.n_quantiles
            )

        return F.sigmoid(output) if self.sigmoid_output else output

    def multi_step_ahead_forward(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ):
        pass

    def forward(self, input_batch: torch.Tensor, target_batch: torch.Tensor):
        if self.one_step_ahead:
            return self.one_step_ahead_forward(
                input_batch=input_batch, target_batch=target_batch
            )
        else:
            return self.multi_step_ahead_forward(
                input_batch=input_batch, target_batch=target_batch
            )


if __name__ == "__main__":
    bs = 16
    t = 10
    h = 60
    w = 60

    input_data = torch.ones(bs, t, h, w)
    for i in range(5):
        input_data[:, i, :, :] = input_data[:, i, :, :] * i
    target_data = torch.ones(bs, t, h, w)
    for i in range(4, 9):
        target_data[:, i - 4, :, :] = target_data[:, i - 4, :, :] * i
    gt_data = torch.ones(bs, t, h, w)
    for i in range(5, 10):
        gt_data[:, i - 5, :, :] = gt_data[:, i - 5, :, :] * i

    model = LSTMSeq2SeqAttn(
        feature_size=h * w,
        linear_output_size=1024,
        lstm_hidden_size=1024,
        lstm_num_layers=6,
        one_step_ahead=True,
        use_cuda=True,
        dropout=0.2,
    )
    for i in tqdm(range(10)):
        preds = model(input_data.view(bs, t, -1), target_data.view(bs, t, -1))
        loss = F.mse_loss(preds, gt_data.view(bs, t, -1))
        loss.backward()

