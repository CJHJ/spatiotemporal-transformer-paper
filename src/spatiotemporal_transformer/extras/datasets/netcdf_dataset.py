from typing import Any, Dict
from pathlib import PurePosixPath

from kedro.io import AbstractVersionedDataSet, Version
from kedro.io.core import get_filepath_str, get_protocol_and_path

import fsspec
import numpy as np
import netCDF4
from netCDF4 import Dataset


class NetCDFDataSet(AbstractVersionedDataSet):
    def __init__(self, filepath: str, attr_name: str, version: Version = None):
        """Creates a new instance of NetCDFDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
            version: The version of the dataset being saved and loaded.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)
        self._format = "NETCDF4"
        self._attr_name = attr_name

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _load(self) -> np.ndarray:
        """Loads data from the netCDF file.

        Returns:
            Data from the netCDF file as numpy array
        """
        load_path = self._get_load_path()
        data = Dataset(load_path, "r", format=self._format)
        data = data.variables[self._attr_name][:].data

        return np.asarray(data)

    def _save(self, data: np.ndarray) -> None:
        """Saves NetCDF data to the specified filepath.
        """
        save_path = self._get_save_path()
        out_data: netCDF4._netCDF4.Dataset = Dataset(
            save_path, "w", format=self._format
        )

        # Create dimensions
        lat_length: int
        lon_length: int
        _, lat_length, lon_length = data.shape
        lat_dim = out_data.createDimension("lat", lat_length)
        lon_dim = out_data.createDimension("lon", lon_length)
        time_dim = out_data.createDimension("time", None)

        # Attributes
        out_data.title = "NetCDF data"

        # Variables
        lat = out_data.createVariable("lat", np.float32, ("lat",))
        lon = out_data.createVariable("lon", np.float32, ("lon",))
        time = out_data.createVariable("time", np.float32, ("time",))
        unit = out_data.createVariable(
            "unit", np.float32, ("time", "lat", "lon")
        )

        # Writing data
        n_lats, n_lons, n_times = len(lat_dim), len(lon_dim), 3
        lat[:] = -90.0 + (180 / n_lats) * np.arange(n_lats)
        lon[:] = (180 / n_lats) * np.arange(n_lons)
        unit[:, :, :] = data

        # TODO: time dimension, general netCDF4 class
        out_data.close()

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.
        """
        return dict(
            filepath=self._filepath,
            version=self._version,
            protocol=self._protocol,
        )
