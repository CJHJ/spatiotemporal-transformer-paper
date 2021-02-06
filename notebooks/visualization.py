# %%
import pathlib
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import pickle

import sys

sys.path.append("../src/")
sys.path.append("./")
import spatiotemporal_transformer.pipelines.data_science.model_main.vis_utils as vu

from metric_calculations import (
    CMAP_PATH,
    MM_PATH,
    NORMAL_MODELS,
    QUANTILE_MODELS,
    PP_SAMPLE_PATH,
    load_and_eval_models,
)

# %%
# Load normal samples
def load_and_package_sample(
    model_name: str, load_path: pathlib.Path, sample_idx: int
) -> Tuple[str, np.ndarray]:
    sample = pickle.load(open(load_path, "rb"))
    if type(sample) == torch.Tensor:
        sample = sample.detach().cpu().numpy()
    sample = sample[sample_idx]
    if len(sample.shape) == 2:
        ts, dim = sample.shape
        wh = int(np.sqrt(dim))
        sample = sample.reshape(ts, wh, wh)

    if model_name == "SpatiotemporalTransformerSeq2Seq":
        model_name = "SpaTrans"
    elif model_name == "VanillaLSTM":
        model_name = "LSTM"
    elif model_name == "VanillaLSTMSeq2Seq":
        model_name = "LSTMS2S"
    elif model_name == "VanillaLSTMSeq2SeqAttn":
        model_name = "LSTMS2SAttn"
    temp_data = (model_name, sample)

    return temp_data


def load_samples(
    model_list: List[str],
    dataset_name: str,
    model_idx: int = 0,
    batch_idx: int = 0,
    sample_idx: int = 0,
) -> List[Tuple[str, np.ndarray]]:
    samples = []
    for i, model_name in enumerate(model_list):
        load_path: pathlib.Path = (
            (MM_PATH if dataset_name == "MM" else CMAP_PATH)
            / f"{model_name}_{model_idx}"
            / (PP_SAMPLE_PATH)
        )

        sample_fn_template = f"{batch_idx}_"
        if i == 0:
            samples.append(
                load_and_package_sample(
                    model_name="Ground Truth",
                    load_path=load_path / (sample_fn_template + "gt.pkl"),
                    sample_idx=sample_idx,
                )
            )
        samples.append(
            load_and_package_sample(
                model_name=model_name,
                load_path=load_path / (sample_fn_template + "preds.pkl"),
                sample_idx=sample_idx,
            )
        )

    return samples


# %%
mm_samples = load_samples(
    model_list=NORMAL_MODELS, dataset_name="MM", batch_idx=0, sample_idx=0
)
cmap_samples = load_samples(
    model_list=NORMAL_MODELS, dataset_name="CMAP", batch_idx=0, sample_idx=0
)

# %%
# vu.plot_diff_mean(sample_data)

# %%
vu.plot_samples(
    mm_samples,
    pred_start_idx=0,
    min_val=0,
    max_val=1,
    wspace=0.1,
    hspace=0.1,
    hsize=20,
    wsize=30,
    model_name_pos=-4.1,
    save_path=Path("./data/mm_0.pdf"),
    gap=2,
)
# %%
vu.plot_samples(
    cmap_samples,
    pred_start_idx=0,
    min_val=0,
    max_val=0.4,
    wspace=0.1,
    hspace=0.1,
    hsize=50,
    wsize=30,
    model_name_pos=-4.1,
    save_path=Path("./data/cmap_0.pdf"),
    gap=2,
)
