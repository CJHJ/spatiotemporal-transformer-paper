import numpy as np


def crop_center(data: np.ndarray) -> np.ndarray:
    width: int
    height: int
    _, width, height = data.shape
    edge_length = int(np.abs(width - height) / 2)

    start_idx = edge_length
    end_idx = start_idx + min([width, height])

    if width > height:
        return data[:, start_idx:end_idx, :]
    return data[:, :, start_idx:end_idx]


def preprocess_cmap(cmap_data: np.ndarray,) -> np.ndarray:
    """Preprocess CMAP data.
    
        Args:
            cmap_data: CMAP data.
        Returns:
            Preprocessed data.
    """
    data = crop_center(cmap_data)

    return data

