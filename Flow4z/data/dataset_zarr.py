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
    file_type: str = 'image'
    def __post_init__(self):
        self.zarr_store = zarr.open_group(self.data_dir, mode="r")
        self.metadata_store = zarr.open_group(self.metadata_dir, mode="r")

    def __len__(self):
        """
        Returns the number of data files available in the dataset directory.
        """
        return len([k for k in self.zarr_store.keys() if k.startswith('data')])

    def _load_image(self, i: int, 
                    sbp: bool) -> tuple[torch.FloatTensor, float]:
        """
        Load an image stamp from file and return it as a tensor along with the maximum value of the stamp.

        Args:
            i (int): Data index.

        Returns:
            torch.FloatTensor: The loaded image stamp.
            float: The maximum value of the stamp.
        """
        stamp = self.zarr_store[f"data_{i}"][:]
        stamp = np.nan_to_num(stamp)
        max_stamp = np.max(stamp, axis=(2,3))
        stamp = torch.FloatTensor(stamp)

        return stamp, max_stamp
    
    def _load_features(self, i: int) -> torch.FloatTensor:
        """
        Load an image stamp from file and return it as a tensor along with the maximum value of the stamp.

        Args:
            i (int): Data index.

        Returns:
            torch.FloatTensor: The loaded image stamp.
            float: The maximum value of the stamp.
        """
        features = self.zarr_store[f"data_{i}"][:]
        features = np.nan_to_num(features)
        features = torch.FloatTensor(features)

        return features

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
        if metadata.shape[2] > 3:
            return metadata[:,:, 0], metadata[:,:, 1], metadata[:,:, 2], metadata[:,:,3]  # z, f, zp, max
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
        if self.file_type == 'image':
            stamps = torch.zeros(size=(len(self.bands), self.nexp, *self.stamp_shape))
            max_norms = torch.zeros(size=(len(self.bands), 1))
            stamps, max_norms = self._load_image(i, self.sbp)
            max_norm = np.mean(max_norms, axis=1)
            stamps = stamps/ max_norm[:,None,None,None]
            meta= torch.Tensor(np.array(self._load_metadata(i, self.sbp)))

            return meta, stamps, max_norm
            
        elif self.file_type == 'features':

            features = self._load_features(i)
            meta= torch.Tensor(np.array(self._load_metadata(i, self.sbp)))
            if meta.shape[0] < 4:
                raise ValueError("max_norm must be stored in metadata (4th column)")
            max_norm = meta[3]
            
            return meta, features, max_norm
        
        