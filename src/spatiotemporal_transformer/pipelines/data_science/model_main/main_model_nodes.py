import os
import logging
import time
from typing import Any, Dict, List, Tuple, Union
import pickle
from pathlib import Path

import numpy as np
from numpy import inf
import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.nn.modules.loss import MSELoss
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from tqdm import tqdm

import pytorch_warmup as warmup

import mlflow

from spatiotemporal_transformer.pipelines.data_science.models.losses import (
    QuantileLoss,
)
from spatiotemporal_transformer.pipelines.data_science.model_main.early_stopping import (
    EarlyStopping,
)

# TODO: Change model here. Need a better model management.
from spatiotemporal_transformer.pipelines.data_science.models.lstm import (
    VanillaLSTM,
)
from spatiotemporal_transformer.pipelines.data_science.models.lstm_seq2seq import (
    LSTMSeq2Seq,
)
from spatiotemporal_transformer.pipelines.data_science.models.lstm_seq2seq_attn import (
    LSTMSeq2SeqAttn,
)
from spatiotemporal_transformer.pipelines.data_science.models.transformer import (
    Transformer,
)
from spatiotemporal_transformer.pipelines.data_science.models.spatiotemporal_transformer_seq2seq import (
    SpatiotemporalTransformerSeq2Seq,
)
from spatiotemporal_transformer.pipelines.data_science.models.convlstm import (
    ConvLSTM,
)

log = logging.getLogger(__name__)

# Keys
QUANTILES = "quantiles"


def prepare_model(
    dataset: Dataset[torch.Tensor],
    dataloader: DataLoader,
    epochs: int,
    model_type: str,
    model_params: Dict,
    optim_params: Dict,
    model_path: Path = None,
) -> Union[nn.Module, Tuple[nn.Module, torch.optim.AdamW, Any, Any]]:
    """Prepare model for training.

    Args:
        dataset: Dataset for model reference.
        dataloader: Dataloader for getting batch step.
        model_params: Model parameters.
        optim_params: Optimizer parameters.
    Returns:
        model: Prepared model for training.
        optimizer: Optimizer for training.
        lr_scheduler: LR scheduler
        warmup_scheduler: Warm up step scheduler.

    """
    model: nn.Module = nn.Module()
    input_width: int = 0
    input_height: int = 0
    use_cuda = model_params["use_cuda"]
    time_length = dataset[0].shape[0]
    feature_size = dataset[0].shape[1]
    if model_type in [
        "SpatiotemporalTransformerSeq2Seq",
    ]:
        input_height = dataset[0].shape[1]
        input_width = dataset[0].shape[2]

    if model_type == "Transformer":
        model = Transformer(
            input_size=feature_size,
            target_size=feature_size,
            time_length=time_length // 2,
            **model_params,
        )
    elif model_type == "VanillaLSTM":
        model = VanillaLSTM(feature_size=feature_size, **model_params)
    elif model_type == "VanillaLSTMSeq2Seq":
        model = LSTMSeq2Seq(feature_size=feature_size, **model_params)
    elif model_type == "VanillaLSTMSeq2SeqAttn":
        model = LSTMSeq2SeqAttn(feature_size=feature_size, **model_params)
    elif model_type == "SpatiotemporalTransformerSeq2Seq":
        model = SpatiotemporalTransformerSeq2Seq(
            input_width=input_width,
            input_height=input_height,
            time_length=time_length // 2,
            **model_params,
        )
    elif model_type == "ConvLSTM":
        model = ConvLSTM(**model_params)

    if use_cuda:
        model = model.cuda()

    if model_path is None:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Parameter processing, division
        optim_params["eps"] = float(optim_params["eps"])
        lr_params = {
            "warmup": optim_params["warmup"],
            "T_max_epoch": optim_params["T_max_epoch"],
            "lr_min": optim_params["lr_min"],
        }
        for key in lr_params.keys():
            del optim_params[key]

        # Initialize optimizer, lr and warmup scheduler
        lr_scheduler = None
        warmup_scheduler = None
        optimizer = torch.optim.AdamW(model.parameters(), **optim_params)

        # Warmup learning rate
        # if model_type in ["SpatiotemporalTransformerSeq2Seq", "Transformer"]:
        if lr_params["warmup"]:
            num_steps = len(dataloader) * lr_params["T_max_epoch"]
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_steps, eta_min=lr_params["lr_min"]
            )
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        return model, optimizer, lr_scheduler, warmup_scheduler
    else:
        if model_path is not None:
            if model_type in [
                "ConvLSTM",
            ]:
                model(
                    dataset[:5][:, : time_length // 2].cuda(),
                    dataset[:5][:, : time_length // 2].cuda(),
                )

        model.load_state_dict(torch.load(model_path).state_dict())
        model.eval()

        return model


def train_model(
    train_dataset: Dataset[torch.Tensor],
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model_type: str,
    model_params: Dict[str, Any],
    optim_params: Dict[str, Any],
    train_params: Dict[str, Any],
):
    """Train model.

    Args:
        train_dataset: Dataset for feature size inference.
        train_dataloader: Dataloader for training data.
        valid_dataloader: Dataloader for validation data.
        test_dataloader: Dataloader for test data.
        model_params: Model parameters.
        optim_params: Optimizer parameters.
        train_params: Training parameters.
    Return:
        Final model pickle file.
    """
    # Training parameters
    epochs: int = train_params["epochs"]
    input_len: int = train_params["input_len"]
    target_len: int = train_params["target_len"]
    test_every: int = train_params["test_every"]
    save_every: int = train_params["save_every"]

    # Quantiles initialization
    loss_fn: nn.Module = nn.Module()
    model_params[QUANTILES] = optim_params[QUANTILES]
    del optim_params[QUANTILES]
    if QUANTILES in model_params and model_params[QUANTILES]:
        loss_fn = QuantileLoss(model_params[QUANTILES])
    else:
        loss_fn = MSELoss()  # Mean over a 2D feature maps

    # Prepare model
    model, optimizer, lr_scheduler, warmup_scheduler = prepare_model(
        train_dataset,
        train_dataloader,
        epochs,
        model_type,
        model_params,
        optim_params,
    )
    if model_params["use_cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Start training regime
    val_loss_fn = {"mse": nn.MSELoss()}
    test_loss_fns = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}
    model = run_experiment(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        input_len=input_len,
        target_len=target_len,
        model_type=model_type,
        model_params=model_params,
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        test_loss_fns=test_loss_fns,
        device=device,
    )

    return model


def run_experiment(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    test_dataloader: DataLoader,
    input_len: int,
    target_len: int,
    model_type: str,
    model_params: Dict[str, Any],
    epochs: int,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    lr_scheduler: torch.optim.lr_scheduler,
    warmup_scheduler,
    loss_fn: nn.Module,
    val_loss_fn: Dict[str, nn.Module],
    test_loss_fns: Dict[str, nn.Module],
    device: torch.device,
):
    early_stopper = EarlyStopping(
        patience=10, verbose=True, delta=0, trace_func=print
    )
    for epoch in range(epochs):
        # Train
        train_loss = train(
            epoch=epoch,
            dataloader=train_dataloader,
            input_len=input_len,
            target_len=target_len,
            model_type=model_type,
            model_params=model_params,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            warmup_scheduler=warmup_scheduler,
            loss_fn=loss_fn,
            device=device,
        )

        # Validation
        val_losses = validate(
            prefix="val",
            epoch=epoch,
            dataloader=valid_dataloader,
            input_len=input_len,
            target_len=target_len,
            model_type=model_type,
            model_params=model_params,
            model=model,
            loss_fns=val_loss_fn,
            device=device,
            save_sample=False,
        )

        # Early stopping and test
        early_stopper(val_loss=val_losses["mse"], epoch=epoch, model=model)
        if early_stopper.early_stop:
            print(f"Patience reached. Test and stopping...")
            break

    # Testing
    test_losses = validate(
        prefix="test",
        epoch=epoch,
        dataloader=test_dataloader,
        input_len=input_len,
        target_len=target_len,
        model_type=model_type,
        model_params=model_params,
        model=model,
        loss_fns=test_loss_fns,
        device=device,
        save_sample=True,
    )

    return model


def predict(
    batch: torch.Tensor,
    input_len: int,
    target_len: int,
    model_type: str,
    model_params: Dict[str, Any],
    model: nn.Module,
    device: torch.device,
):
    # Batch data
    batch = batch.to(device)
    input_batch = batch[:, :input_len]
    ground_truth_batch = batch[:, input_len : input_len + target_len]
    target_batch: Union[torch.Tensor, None] = None
    if model_type in [
        "Transformer",
        "SpatiotemporalTransformerSeq2Seq",
        "VanillaLSTMSeq2Seq",
        "VanillaLSTMSeq2SeqAttn",
    ]:
        target_batch = batch[:, input_len - 1 : -1]

    # Mask data
    input_mask: Union[torch.Tensor, None] = None
    target_mask: Union[torch.Tensor, None] = None
    if model_type == "Transformer":
        input_mask = torch.ones(1, input_len, input_len).to(device)
        target_mask = np.triu(
            np.ones((1, target_len, target_len)), k=1
        ).astype(np.int8)
        target_mask = (torch.from_numpy(target_mask) == 0).to(device)
    elif model_type == "SpatiotemporalTransformerSeq2Seq":
        height, width = batch.shape[2:]
        num_patch = (width // model_params["patch_size"]) * (
            height // model_params["patch_size"]
        )
        input_mask = torch.ones(
            1,
            input_len,
            input_len,
            model_params["num_heads"],
            num_patch,
            num_patch,
        ).to(device)
        target_mask = np.triu(
            np.ones((1, target_len, target_len)), k=1
        ).astype(np.int8)
        target_mask = (torch.from_numpy(target_mask) == 0).to(device)
        target_mask = torch.einsum(
            "b t u, n h w -> b t u n h w",
            target_mask,
            torch.ones((model_params["num_heads"], num_patch, num_patch)).to(
                device
            ),
        )

    # Predict using model
    preds: torch.Tensor = torch.Tensor()
    scale: torch.Tensor = torch.Tensor()
    if model_type in [
        "Transformer",
        "SpatiotemporalTransformerSeq2Seq",
    ]:
        preds = model(input_batch, target_batch, input_mask, target_mask)
    elif model_type in ["VanillaLSTM"]:
        preds = model(batch, input_len, target_len)
    elif model_type == "ConvLSTM":
        preds = model(batch, ground_truth_batch)
    elif model_type in [
        "VanillaLSTMSeq2Seq",
        "VanillaLSTMSeq2SeqAttn",
    ]:
        preds = model(input_batch, target_batch)

    return preds, ground_truth_batch, scale


def calculate_loss(
    preds: torch.Tensor, ground_truth: torch.Tensor, loss_fn: nn.Module
):
    loss = loss_fn(preds, ground_truth)

    return loss


def optimize(
    optimizer: torch.optim.AdamW,
    preds: torch.Tensor,
    ground_truth: torch.Tensor,
    loss_fn: nn.Module,
):
    optimizer.zero_grad()
    loss = calculate_loss(
        preds=preds, ground_truth=ground_truth, loss_fn=loss_fn
    )
    loss.backward()
    optimizer.step()

    return loss


def train(
    epoch: int,
    dataloader: DataLoader,
    input_len: int,
    target_len: int,
    model_type: str,
    model_params: Dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    lr_scheduler,
    warmup_scheduler,
    loss_fn: nn.Module,
    device: torch.device,
):
    total_loss = 0
    n_total = 0
    model.train(True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Calculate loss and optimize accordingly
        # Prediction
        preds, ground_truth_batch, _ = predict(
            batch=batch,
            input_len=input_len,
            target_len=target_len,
            model_type=model_type,
            model_params=model_params,
            model=model,
            device=device,
        )
        loss = optimize(
            optimizer=optimizer,
            preds=preds,
            ground_truth=ground_truth_batch,
            loss_fn=loss_fn,
        )

        if lr_scheduler != None:
            lr_scheduler.step()
            warmup_scheduler.dampen()

        # Total loss
        total_loss += float(loss.item()) * batch.shape[0]
        n_total += batch.shape[0]

    # Average loss
    loss_avg = total_loss / n_total
    log.info(f"epoch = {epoch}, loss = {loss_avg}")
    mlflow.log_metric("train_mse", loss_avg, epoch + 1)

    return loss_avg


def validate(
    prefix: str,
    epoch: int,
    dataloader: DataLoader,
    input_len: int,
    target_len: int,
    model_type: str,
    model_params: Dict[str, Any],
    model: nn.Module,
    loss_fns: Dict[str, nn.Module],
    device: torch.device,
    save_sample: bool,
):
    total_test_losses = {}
    avg_test_losses = {}
    for key in loss_fns.keys():
        total_test_losses[key] = 0
    n_test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Prediction
            preds, ground_truth_batch, scale = predict(
                batch=batch,
                input_len=input_len,
                target_len=target_len,
                model_type=model_type,
                model_params=model_params,
                model=model,
                device=device,
            )

            # Loss calculation
            if QUANTILES in model_params and model_params[QUANTILES]:
                median_idx = model_params[QUANTILES].index(0.5)
                preds = preds[:, :, median_idx]

            for key in loss_fns.keys():
                loss = calculate_loss(
                    preds, ground_truth_batch, loss_fn=loss_fns[key]
                )
                total_test_losses[key] += float(loss.item()) * batch.shape[0]
            n_test_loss += batch.shape[0]

            # Save sample
            if save_sample:
                path_artifact_local = Path("temp_artifact")
                path_sample_artifact = path_artifact_local / "samples"
                os.makedirs(path_sample_artifact, exist_ok=True)
                if batch_idx in [0, 1, 2]:
                    pickle.dump(
                        ground_truth_batch.detach().cpu().numpy(),
                        open(
                            path_sample_artifact / f"{batch_idx}_gt.pkl", "wb",
                        ),
                    )
                    pickle.dump(
                        preds.detach().cpu().numpy(),
                        open(
                            path_sample_artifact / f"{batch_idx}_preds.pkl",
                            "wb",
                        ),
                    )

                # Move to respective artifact location
                mlflow.log_artifact(
                    path_artifact_local, artifact_path="artifacts"
                )

        # Average losses
        for key in loss_fns.keys():
            avg_test_losses[key] = total_test_losses[key] / n_test_loss
            log.info(
                f"loss: {key}, epoch = {epoch}, loss = {avg_test_losses[key]}"
            )

            # MLFlow log metrics
            mlflow.log_metric(
                f"{prefix}_loss_{key}", avg_test_losses[key], epoch + 1,
            )

    return avg_test_losses
