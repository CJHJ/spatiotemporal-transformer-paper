# %%
import copy
from pathlib import Path
import sys
import pickle
from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn.modules.loss import MSELoss
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from tqdm import tqdm


sys.path.append("../src/")
sys.path.append("../src/spatiotemporal_transformer/pipelines/")
sys.path.append(
    "../src/spatiotemporal_transformer/pipelines/data_science/models"
)

from data_science.model_input.nodes import (
    split_data,
    make_time_series_data,
    prepare_scale_data,
    init_scaler,
    normalize_data,
)
from data_science.model_main.data_utils_nodes import (
    create_dataloader,
    create_dataset,
)
from data_science.model_main.main_model_nodes import prepare_model

from spatiotemporal_transformer.pipelines.data_science.models.losses import (
    ICP,
    MIL,
    QuantileLoss,
    convert_gaussian_to_quantiles,
)
from spatiotemporal_transformer.pipelines.data_science.model_main.main_model_nodes import (
    calculate_loss,
    predict,
)
from spatiotemporal_transformer_seq2seq import SpatiotemporalTransformerSeq2Seq
from transformer import Transformer
from lstm import VanillaLSTM
from lstm_seq2seq import LSTMSeq2Seq
from lstm_seq2seq_attn import LSTMSeq2SeqAttn
from convlstm import ConvLSTM
from convlstm_seq2seq import ConvLSTMSeq2Seq
from convlstm_seq2seq_attn import ConvLSTMSeq2SeqAttn
from cnmm import CNMM
from dmm import DMM

from spatiotemporal_transformer.extras.params_ref import (
    QUANTILES_OPTIM_PARAMS,
    OPTIM_PARAMS,
    FLATTEN_MODELS,
    MODEL_PARAMS,
)

PYRO_MODELS = ["DMM", "CNMM"]
QUANTILES = "quantiles"

# %%
# Paths
CMAP_PATH = Path("../models/CMAP")
MM_PATH = Path("../models/MovingMNIST")

NORMAL_MODELS = [
    "SpatiotemporalTransformerSeq2Seq",
    "Transformer",
    "VanillaLSTM",
    "VanillaLSTMSeq2Seq",
    "VanillaLSTMSeq2SeqAttn",
    "ConvLSTM",
    "ConvLSTMSeq2Seq",
    "ConvLSTMSeq2SeqAttn",
]
QUANTILE_MODELS = ["quantiles_" + model for model in NORMAL_MODELS] + [
    "CNMM",
    "DMM",
]

PP_SAMPLE_PATH = Path("artifacts/temp_artifact/samples")
MODEL_PATH = Path("models/data/model.pth")

# %%
# Loading data
def load_cmap_data(
    seq_len: int = 20,
    scaler_type: str = "MinMaxScaler",
    min_max_vals: List[int] = [0, 170],
):
    scaler_params = {"scaler_type": scaler_type, "min_max_vals": min_max_vals}
    cmap_data = pickle.load(
        open("../data/02_intermediate/preprocessed_cmap.pkl", "rb")
    )
    train_data, valid_data, test_data = split_data(
        cmap_data, {"train_valid_size": 0.8, "train_size": 0.8}
    )
    train_series_data = make_time_series_data(seq_len, train_data)
    valid_series_data = make_time_series_data(seq_len, valid_data)
    test_series_data = make_time_series_data(seq_len, test_data)
    data_for_fitting = prepare_scale_data(train_data, scaler_params)
    scaler = init_scaler(scaler_params, data_for_fitting)
    train_data = normalize_data(scaler, train_series_data)
    valid_data = normalize_data(scaler, valid_series_data)
    test_data = normalize_data(scaler, test_series_data)

    return train_data, valid_data, test_data, scaler


def load_mm_data(
    scaler_type: str = "MinMaxScaler", min_max_vals: List[int] = [0, 255],
):
    scaler_params = {"scaler_type": scaler_type, "min_max_vals": min_max_vals}
    mm_data = pickle.load(open("../data/01_raw/mnist_test_seq.pkl", "rb"))
    train_data, valid_data, test_data = split_data(
        mm_data, {"train_valid_size": 0.8, "train_size": 0.8}
    )
    data_for_fitting = prepare_scale_data(train_data, scaler_params)
    scaler = init_scaler(scaler_params, data_for_fitting)
    train_data = normalize_data(scaler, train_data)
    valid_data = normalize_data(scaler, valid_data)
    test_data = normalize_data(scaler, test_data)

    return train_data, valid_data, test_data, scaler


# %%
# Load and eval models
def load_and_eval_models(
    n_models: int = 5,
    input_length: int = 10,
    target_length: int = 10,
    data_type: str = "MM",
    quantiles: bool = True,
    dataset: Dict[str, Dataset] = {"unflattened": None, "flattened": None,},
    dataloader: Dict[str, DataLoader] = {
        "unflattened": None,
        "flattened": None,
    },
    reported_loss_save_path: Path = None,
):
    data_path = CMAP_PATH if data_type == "CMAP" else MM_PATH
    model_list = QUANTILE_MODELS if quantiles else NORMAL_MODELS
    reported_losses = {}
    for model_type in model_list:
        print(f"Model type: {model_type}")
        clean_model_type = model_type.replace("quantiles_", "")
        flatten = (
            "flattened"
            if clean_model_type in FLATTEN_MODELS
            else "unflattened"
        )
        temp_dataset = dataset[flatten]
        temp_dataloader = dataloader[flatten]
        reported_losses[clean_model_type] = {}
        for model_number in range(n_models):
            print(f"Model_number: {model_number}")

            # Create load path
            model_artifact_path = data_path / (model_type + f"_{model_number}")
            sample_path = model_artifact_path / PP_SAMPLE_PATH
            model_path = model_artifact_path / MODEL_PATH

            # Clean model_type and load model and optim params
            model_params = copy.copy(MODEL_PARAMS[clean_model_type])
            optim_params = (
                copy.copy(QUANTILES_OPTIM_PARAMS)
                if quantiles
                else OPTIM_PARAMS
            )

            # Quantiles initialization
            if model_type not in PYRO_MODELS:
                model_params[QUANTILES] = optim_params[QUANTILES]

            # Load model
            model = prepare_model(
                dataset=temp_dataset,
                dataloader=temp_dataloader,
                epochs=0,
                model_type=clean_model_type,
                model_params=model_params,
                optim_params=optim_params,
                model_path=model_path,
            )
            if model_params["use_cuda"]:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Evaluate model
            loss_fns = {
                "QuantileLoss": QuantileLoss(optim_params[QUANTILES]),
                "IntervalCoveragePercentage": ICP(optim_params[QUANTILES]),
                "MeanIntervalLength": MIL(optim_params[QUANTILES]),
            }
            total_test_losses = {}
            avg_test_losses = {}
            for key in loss_fns.keys():
                total_test_losses[key] = 0
            n_test_loss = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(temp_dataloader)):
                    # Prediction
                    preds, ground_truth_batch, scale = predict(
                        batch=batch,
                        input_len=input_length,
                        target_len=target_length,
                        model_type=clean_model_type,
                        model_params=model_params,
                        model=model,
                        device=device,
                    )

                    # Calculate losses
                    if clean_model_type in PYRO_MODELS:
                        preds = convert_gaussian_to_quantiles(
                            optim_params[QUANTILES], preds, scale
                        )
                    for key in loss_fns.keys():
                        loss = loss_fns[key](preds, ground_truth_batch)
                        total_test_losses[key] += (
                            float(loss.item()) * batch.shape[0]
                        )
                    n_test_loss += batch.shape[0]

            # Average losses
            for key in loss_fns.keys():
                avg_test_losses[key] = total_test_losses[key] / n_test_loss
                print(f"{key} loss = {avg_test_losses[key]}")
                if key not in reported_losses[clean_model_type].keys():
                    reported_losses[clean_model_type][key] = []
                reported_losses[clean_model_type][key].append(
                    avg_test_losses[key]
                )

    if reported_loss_save_path is not None:
        pickle.dump(reported_losses, open(reported_loss_save_path, "wb"))


# %%
# Data loader
def make_dataset(data, flatten: bool):
    return create_dataset(data, {"flatten": flatten})


def make_dataloader(
    data,
    params: Dict[str, Any] = {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 4,
    },
):
    return create_dataloader(data, params)


def make_dataset_and_dataloader(data):
    flatten_dataset = make_dataset(data, flatten=True)
    flatten_dataloader = make_dataloader(flatten_dataset)

    unflatten_dataset = make_dataset(data, flatten=False)
    unflatten_dataloader = make_dataloader(unflatten_dataset)

    return (
        flatten_dataset,
        flatten_dataloader,
        unflatten_dataset,
        unflatten_dataloader,
    )


if __name__ == "__main__":

    mm_train_data, mm_valid_data, mm_test_data, mm_scaler = load_mm_data()
    (
        cmap_train_data,
        cmap_valid_data,
        cmap_test_data,
        cmap_scaler,
    ) = load_cmap_data()
    print("CMAP")
    print(
        f"Shapes: train: {cmap_train_data.shape}, valid: {cmap_valid_data.shape},test: {cmap_test_data.shape}"
    )
    print(
        f"Sums: train: {cmap_train_data.sum()}, valid: {cmap_valid_data.sum()},test: {cmap_test_data.sum()}"
    )
    print("MM")
    print(
        f"train: {mm_train_data.shape}, valid: {mm_valid_data.shape},test: {mm_test_data.shape}"
    )
    print(
        f"Sums: train: {mm_train_data.sum()}, valid: {mm_valid_data.sum()},test: {mm_test_data.sum()}"
    )

    (
        mm_flat_ds,
        mm_flat_dl,
        mm_unflat_ds,
        mm_unflat_dl,
    ) = make_dataset_and_dataloader(mm_test_data)
    (
        cmap_flat_ds,
        cmap_flat_dl,
        cmap_unflat_ds,
        cmap_unflat_dl,
    ) = make_dataset_and_dataloader(cmap_test_data)

    #%%
    mm_flat_ds[0].shape

    load_and_eval_models(
        n_models=5,
        data_type="MM",
        quantiles=True,
        dataset={"unflattened": mm_unflat_ds, "flattened": mm_flat_ds,},
        dataloader={"unflattened": mm_unflat_dl, "flattened": mm_flat_dl,},
        reported_loss_save_path=Path("./mm_report_transformer.pkl"),
    )
    load_and_eval_models(
        n_models=5,
        data_type="CMAP",
        quantiles=True,
        dataset={"unflattened": cmap_unflat_ds, "flattened": cmap_flat_ds,},
        dataloader={"unflattened": cmap_unflat_dl, "flattened": cmap_flat_dl,},
        reported_loss_save_path=Path("./cmap_transformer.pkl"),
    )

    # %%
    load_and_eval_models(
        n_models=5,
        data_type="MM",
        quantiles=False,
        dataset={"unflattened": mm_unflat_ds, "flattened": mm_flat_ds,},
        dataloader={"unflattened": mm_unflat_dl, "flattened": mm_flat_dl,},
    )
    load_and_eval_models(
        n_models=5,
        data_type="CMAP",
        quantiles=False,
        dataset={"unflattened": cmap_unflat_ds, "flattened": cmap_flat_ds,},
        dataloader={"unflattened": cmap_unflat_dl, "flattened": cmap_flat_dl,},
    )

    # %%
