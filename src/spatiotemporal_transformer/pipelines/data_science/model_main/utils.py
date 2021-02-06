from typing import Union

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch
import torch.nn as nn
from spatiotemporal_transformer.pipelines.data_science.model_input.nodes import (
    denormalize_data,
)


# Multistep forecast helper functions
def multistep_horizon_forecast(
    model: nn.Module,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    input_mask: torch.Tensor,
    target_mask: torch.Tensor,
    device: str,
):
    # Target batch timestep + 1
    target_batch_shape = list(target_batch.shape)
    target_batch_shape[1] += 1

    # Initialize target batch with first timestep target_batch
    multi_horizon_target_batch = torch.zeros(target_batch_shape).to(device)
    multi_horizon_target_batch[:, 0] = target_batch[:, 0]
    for t in range(target_batch.shape[1]):  # Predict one by one
        multi_horizon_target_batch[:, t + 1] = model(
            input_batch,
            multi_horizon_target_batch[:, :-1],
            input_mask,
            target_mask,
        )[:, t]

    return multi_horizon_target_batch[:, 1:]


def multi_onestep_horizon_forecast(
    model: nn.Module,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    input_mask: torch.Tensor,
    target_mask: torch.Tensor,
    device: str,
):
    # Initialize target batch with first timestep target_batch
    multi_horizon_target_batch = torch.zeros(target_batch.shape).to(device)
    multi_horizon_preds_batch = torch.zeros(target_batch.shape).to(device)

    # Predict one step multiple times
    for t in range(target_batch.shape[1]):  # Predict one by one
        multi_horizon_target_batch[:, t] = target_batch[:, t]
        multi_horizon_preds_batch[:, t] = model(
            input_batch, multi_horizon_target_batch, input_mask, target_mask,
        )[:, t]

    return multi_horizon_preds_batch


# Loss calculation
def calculate_denorm_loss(
    loss_fn: torch.functional,
    scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
):
    prediction = denormalize_data(scaler, prediction.cpu().detach().numpy())
    ground_truth = denormalize_data(
        scaler, ground_truth.cpu().detach().numpy()
    )

    print(f"Pred: {prediction.shape}, GT: {ground_truth.shape}")

    denormalized_loss = loss_fn(
        torch.from_numpy(prediction), torch.from_numpy(ground_truth)
    )

    return denormalized_loss

