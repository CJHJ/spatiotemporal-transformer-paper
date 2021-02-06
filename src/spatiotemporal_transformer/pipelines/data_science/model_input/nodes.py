from typing import Union, Dict, Tuple
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

log = logging.getLogger(__name__)


def split_data(
    data: np.ndarray, split_params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test data.

        Args:
            data: Original data.
        Returns:
            Split data (train, valid and test).
    """
    train_data: np.ndarray
    valid_data: np.ndarray
    test_data: np.ndarray
    train_data, test_data = train_test_split(
        data, train_size=split_params["train_valid_size"], shuffle=False,
    )
    train_data, valid_data = train_test_split(
        train_data, train_size=split_params["train_size"], shuffle=False,
    )

    return (train_data, valid_data, test_data)


def make_time_series_data(time_length: int, data: np.ndarray):
    """Make a time series data from sequential data.

        Args:
            data: Sequential data
        Returns:
            Time series data.
    """
    train_series_data = []
    data_len = len(data)
    for i in range(data_len):
        if i + time_length <= data_len:
            train_series_data.append(data[i : i + time_length, :, :])

    return np.stack(train_series_data, axis=0)


def prepare_scale_data(
    ref_data: np.ndarray, scaler_params: Dict
) -> np.ndarray:
    """Prepare data for scale fitting.
    
        Args:
            ref_data: Reference data to output as data to be fitted.
            scaler_params: Parameters for scaler. As of now, it only handles optional min max values for MinMaxScaler.
        Returns:
            Data for scale fitting.
    """
    if scaler_params["scaler_type"] == "MinMaxScaler":
        ref_data = np.array(scaler_params["min_max_vals"])

    return np.expand_dims(ref_data.flatten(), axis=1)


def init_scaler(
    scaler_parameters: Dict, fit_data: np.ndarray,
) -> Union[MinMaxScaler, StandardScaler, RobustScaler]:
    """Initialize and return scaler.

        Args:
            scaler_parameters: Parameters of scaler. 
            fit_data: Data to be fit.
        Returns:
            Selected scaler.
    """
    scaler_type = scaler_parameters["scaler_type"]
    if scaler_type == "RobustScaler":
        scaler = RobustScaler()
    elif scaler_type == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaler.fit(fit_data)

    return scaler


def normalize_data(
    scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
    data: np.ndarray,
) -> np.ndarray:
    """Normalize data.
    
        Args:
            scaler: Scaler/Normalizer.
            data: Data to be normalize.
        Returns:
            Normalized data.
    """
    reshaped_data = np.expand_dims(data.flatten(), axis=1)

    return scaler.transform(reshaped_data).reshape(data.shape)


def denormalize_data(
    scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
    data: np.ndarray,
) -> np.ndarray:
    """Denormalize data.
    
        Args:
            scaler: Scaler/Normalizer.
            data: Data to be normalize.
        Returns:
            Denormalized data.
    """
    reshaped_data = np.expand_dims(data.flatten(), axis=1)

    return scaler.inverse_transform(reshaped_data).reshape(data.shape)

