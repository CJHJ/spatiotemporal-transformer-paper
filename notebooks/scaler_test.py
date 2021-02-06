# %%
import numpy as np
from typing import Union, Dict

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


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


# %%
fit_data = np.random.rand(4, 4, 4) * 10


# %%
fit_data
# %%
scaler_params = {
    "scaler_type": "MinMaxScaler",
    "min_max_vals": [0, 10],
}
scale_data = prepare_scale_data(fit_data, scaler_params=scaler_params,)

# %%
scale_data

# %%
scaler = init_scaler(scaler_parameters=scaler_params, fit_data=scale_data)

# %%
scaled_data = normalize_data(scaler, fit_data)

# %%
scaled_data

# %%
returned_data = denormalize_data(scaler, scaled_data)

# %%
returned_data.sum()

# %%
fit_data.sum()

# %%
