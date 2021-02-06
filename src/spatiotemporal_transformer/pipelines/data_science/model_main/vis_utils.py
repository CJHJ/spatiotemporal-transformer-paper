from typing import List, Tuple
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec


def calc_avg(data: np.ndarray):
    """
        Calculate average over 2D features.

        Args:
            data (np.ndarray[time, height, width]): Data

        Return:
            mean (np.ndarray[time]): Averaged data
    """
    return data.mean(axis=(1, 2))


def plot_diff_mean(
    sample_seqs: List[List[Tuple[str, np.ndarray]]],
    n_row: int = 3,
    n_col: int = 3,
    size: int = 10,
    fontsize: int = 18,
    save_path: Path = None,
):
    """Plot spatial squared error difference between ground truth and prediction.

        Args:
            sample_seqs (List[List[Tuple[str, np.ndarray]]]): List of batches -> List of model -> name of the model, data.
            n_row (int): Number of boxes in one row.
            n_col (int): Number of boxes in one column.
            size (int): Size of figure.
            fontsize (int): Fontsize.
            save (Path): Save path.
    """
    font = {"family": "normal", "weight": "normal", "size": fontsize}

    matplotlib.rc("font", **font)
    plt.figure(figsize=(size, size - 2))

    total_box = n_row * n_col
    assert len(sample_seqs) < total_box

    # For every batch
    for i, sample_seq in enumerate(sample_seqs):
        time_length = sample_seq[0][1].shape[0]
        plt.subplot(n_row, n_col, i + 1)

        plt.xlim(0, time_length - 1)
        plt.xlabel("Timestep")
        plt.ylabel("Spatial mean squared error")
        plt.grid(True)

        # For every sample
        for key_data in sample_seq:
            label = key_data[0]
            data_avg = calc_avg(key_data[1])
            print(data_avg.shape)
            plt.semilogy(data_avg, label=label)
        plt.legend()
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_samples(
    sample_seqs: List[Tuple[str, np.ndarray]],
    pred_start_idx: int = None,
    min_val: float = 0,
    max_val: float = 170,
    hsize: int = 140,
    wsize: int = 20,
    wspace: float = 0.1,
    hspace: float = 0.1,
    fontsize: int = 60,
    model_name_pos: float = -2.1,
    gap: int = 1,
    save_path: Path = None,
):
    """Plot heatmap or error difference data in a timeline.

        Args:
            sample_seqs (List[Tuple[str, np.ndarray]]): List of model -> name of the model, data.
            pred_start_idx (int): Start index of prediction. Setting to None will delete the indicator.
            min_val (float): Minimum value of heatmap.
            max_val (float): Maximum value of heatmap.
            hsize (int): Height of figure.
            wsize (int): Width of figure.
            wspace (int): Space width.
            hspace (int): Space height.
            fontsize (int): Fontsize.
            model_name_pos (float): Model name text position.
            gap (int): Timestep gap of visualization. Frames within the gap will be skipped.
            save (Path): Save path.
    """
    font = {"family": "normal", "weight": "normal", "size": fontsize}
    matplotlib.rc("font", **font)

    # Total model to be visualized
    nrow = len(sample_seqs)
    height_ratios = [1 for i in range(nrow)]
    time_length = sample_seqs[0][1].shape[0]

    ncol = time_length // gap
    plt.figure(figsize=(hsize, wsize))

    gs = gridspec.GridSpec(
        nrow,
        ncol,
        height_ratios=height_ratios,
        width_ratios=np.ones(ncol),
        wspace=wspace,
        hspace=hspace,
    )

    for i in range(0, time_length, gap):
        for j, data in enumerate(
            [(sample_seq[0], sample_seq[1][i]) for sample_seq in sample_seqs]
        ):
            label, image = data
            ax = plt.subplot(gs[j, i // gap])
            plt.imshow(
                image,
                cmap="hot",
                interpolation="nearest",
                vmin=min_val,
                vmax=max_val,
            )
            plt.axis("off")
            if i == 0:
                plt.text(model_name_pos, 0.4, label, transform=ax.transAxes)

            if j == nrow - 1:
                plt.text(0.4, -0.4, str(i + 1), transform=ax.transAxes)
                if pred_start_idx and i == (pred_start_idx - 1):
                    plt.text(
                        0.0, -0.8, str("Prediction →"), transform=ax.transAxes,
                    )
            ax.set_aspect("equal")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

