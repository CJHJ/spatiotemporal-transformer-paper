from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PlainDataset(Dataset):
    def __init__(self, data: np.ndarray, dataset_params: Dict):
        self.data = torch.from_numpy(data).float()
        self.flatten: bool = dataset_params["flatten"]

    def __getitem__(self, index) -> torch.Tensor:
        seq_len = self.data.shape[1]
        if self.flatten:
            return self.data[index].reshape((seq_len, -1))
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def create_dataset(data: np.ndarray, dataset_params: Dict) -> Dataset:
    """Create dataset.

        Args:
            data: Train data.
        Returns:
            Dataset. 
    """
    return PlainDataset(data, dataset_params)


def create_dataloader(dataset: Dataset, params: Dict) -> DataLoader:
    """Create dataloader.
    
        Args:
            dataset: Train dataset.
        Returns:
            Dataloader.
    """
    return DataLoader(dataset, **params)
