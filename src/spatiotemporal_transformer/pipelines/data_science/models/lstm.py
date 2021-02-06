"""Vanilla LSTM"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTM
from einops import rearrange


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        feature_size: int,
        linear_output_size: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        one_step_ahead: bool,
        hidden_seq_len: int,
        use_cuda: bool,
        dropout: float,
        quantiles: List[float] = [],
    ):
        super(VanillaLSTM, self).__init__()
        self.hidden_seq_len = hidden_seq_len
        self.one_step_ahead = one_step_ahead
        self.n_quantiles = len(quantiles) if quantiles else 1

        self.linear_in = nn.Linear(feature_size, linear_output_size)
        self.lstm = LSTM(
            linear_output_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear_out = nn.Linear(
            lstm_hidden_size * hidden_seq_len, feature_size * self.n_quantiles
        )

        self.h_0 = nn.Parameter(
            torch.zeros(lstm_num_layers, 1, lstm_hidden_size)
        )
        self.c_0 = nn.Parameter(
            torch.zeros(lstm_num_layers, 1, lstm_hidden_size)
        )

    def one_step_ahead_forward(
        self, x: torch.Tensor, start_pred: int, pred_length: int
    ):
        batch_size = x.shape[0]
        preds = []
        for i in range(pred_length):
            temp = x[:, i : start_pred + i, :]
            temp = F.gelu(self.linear_in(temp))

            h_n = self.h_0.expand(-1, batch_size, -1).contiguous()
            c_n = self.c_0.expand(-1, batch_size, -1).contiguous()
            lstm_output, _ = self.lstm(temp, (h_n, c_n))

            output = self.linear_out(
                lstm_output[:, -self.hidden_seq_len :, :].reshape(
                    batch_size, -1
                )
            )
            preds.append(output.unsqueeze(1))

        preds = torch.cat(preds, dim=1)
        if self.n_quantiles > 1:
            preds = rearrange(
                preds, "b t (q d) -> b t q d", q=self.n_quantiles
            )

        return preds

    def multi_step_ahead_forward(
        self, x: torch.Tensor, start_pred: int, pred_length: int
    ):
        batch_size = x.shape[0]
        preds = []
        for i in range(pred_length):
            if i == 0:
                temp = x[:, i:start_pred, :]
            else:
                temp = torch.cat(
                    [x[:, i:start_pred, :], torch.cat(preds[:i], dim=1)], dim=1
                )
            temp = F.gelu(self.linear_in(temp))

            h_n = self.h_0.expand(-1, batch_size, -1).contiguous()
            c_n = self.c_0.expand(-1, batch_size, -1).contiguous()
            lstm_output, _ = self.lstm(temp, (h_n, c_n))

            output = self.linear_out(
                lstm_output[:, :, :].reshape(batch_size, -1)
            )
            preds.append(output.unsqueeze(1))

        preds = torch.cat(preds, dim=1)
        if self.n_quantiles > 1:
            preds = rearrange(
                preds, "b t (q d) -> b t q d", q=self.n_quantiles
            )

        return preds

    def forward(self, x: torch.Tensor, start_pred: int, pred_length: int):
        if self.one_step_ahead:
            return self.one_step_ahead_forward(x, start_pred, pred_length)
        else:
            return self.multi_step_ahead_forward(x, start_pred, pred_length)

