import numpy as np
from pathlib import Path
import torch

def load_image(data_dir: Path | str, 
               i: int,
               band: str, 
               exp: int = 0) -> tuple[torch.FloatTensor, float, np.ndarray]:
    """
    Load a specific image and its metadata from the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        i (int): Index of the data sample.
        band (str): Photometric band (e.g., 'CFHT_U').
        exp (int): Exposure number (default: 0).

    Returns:
        stamp (torch.FloatTensor): The image data (60x60 pixels).
        max_stamp (float): The maximum pixel value in the stamp.
        meta (np.ndarray): Metadata associated with the image.
    """
    path = Path(data_dir) / f'data_{i}' / f'cutout_{band}_exp{exp}.npy'
    path_meta = Path(data_dir) / f'data_{i}' / f'metadata_{band}_exp{exp}.npy'

    # Load image and metadata
    stamp = np.load(path).reshape(60, 60)
    meta = np.load(path_meta)
    
    # Handle NaNs in the stamp
    stamp = np.nan_to_num(stamp)
    max_stamp = np.max(stamp)
    stamp = torch.FloatTensor(stamp)

    return stamp, max_stamp, meta