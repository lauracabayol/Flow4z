import numpy as np
from pathlib import Path
import torch
import zarr
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class DataSet:
    """
    Custom dataset loader for handling different types of data including images and features.

    Attributes:
        data_dir (Path): Directory containing the dataset.
        bands (list): List of band identifiers.
        multiple_exps (bool): Flag indicating if multiple exposures are present.
        stamp_shape (tuple): Shape of the image stamps.
        nexp (int): Number of exposures.
        zp_calib (bool): Flag indicating if zero-point calibration is used.
        file_type (str): Type of file to load ('image' or 'features').
        size_meta (int): Size of metadata depending on zp_calib.
    """
    data_dir: Path
    metadata_dir: Path
    bands: list
    nexp: int = 3
    stamp_shape: tuple = (60, 60)
    size_meta: int = 3
    sbp: bool = False

    def __post_init__(self):
        self.zarr_store = zarr.open_group(self.data_dir, mode="r")
        self.metadata_store = zarr.open_group(self.metadata_dir, mode="r")

    def __len__(self):
        """
        Returns the number of data files available in the dataset directory.
        """
        return len(list(self.zarr_store.group_keys()))

    def _load_image(self, i: int, 
                    sbp: bool) -> tuple[torch.FloatTensor, float]:
        """
        Load an image stamp from file and return it as a tensor along with the maximum value of the stamp.

        Args:
            i (int): Data index.
            band (str): Band identifier.
            exp (int): Exposure number.

        Returns:
            torch.FloatTensor: The loaded image stamp.
            float: The maximum value of the stamp.
        """
        stamp = self.zarr_store[f"data_{i}"][:]
        stamp = np.nan_to_num(stamp)
        max_stamp = np.max(stamp, axis=(2,3))
        stamp = torch.FloatTensor(stamp)

        if sbp:
            stamp = stamp.reshape(self.nexp*len(self.bands), *self.stamp_shape)
            
        return stamp, max_stamp

    def _load_metadata(self, i: int, sbp: bool) -> tuple[float, float, float]:
        """
        Load metadata from file and return it as a tuple.

        Args:
            i (int): Data index.
            band (str): Band identifier.
            exp (int): Exposure number.

        Returns:
            tuple: z, f, zp values from metadata.
        """
        metadata = self.metadata_store[f"data_{i}"][:]
        if sbp:
            metadata = metadata.reshape(self.nexp*len(self.bands), self.size_meta)
            return metadata[:, 0], metadata[:, 1], metadata[:, 2]  # z, f, zp
        else:
            return metadata[:,:, 0], metadata[:,:, 1], metadata[:,:, 2]  # z, f, zp

    def __getitem__(self, i: int) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Load and normalize images and metadata for multiple exposures.

        Args:
            i (int): Data index.

        Returns:
            tuple: Metadata, image stamps, and maximum norms.
        """
        stamps = torch.zeros(size=(len(self.bands), self.nexp, *self.stamp_shape))
        max_norms = torch.zeros(size=(len(self.bands), 1))
        
        for ib, b in enumerate(self.bands):
            max_norm = 0

        meta= torch.DoubleTensor(self._load_metadata(i, self.sbp))
        stamps, max_norms = self._load_image(i, self.sbp)

        print(meta.shape, stamps.shape, max_norms.shape)

        #stamps[ib] = stamps[ib] / max_norms[ib]
        
        return meta, stamps, max_norms
