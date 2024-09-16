import numpy as np
import os
import torch

def load_image(data_dir, i, band, exp=0):
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
    path = f'{data_dir}/data_{i}/cutout_{band}_exp{exp}.npy'
    path_meta = f'{data_dir}/data_{i}/metadata_{band}_exp{exp}.npy'

    # Load image and metadata
    stamp = np.load(path).reshape(60, 60)
    meta = np.load(path_meta)
    
    # Handle NaNs in the stamp
    stamp = np.nan_to_num(stamp)
    max_stamp = np.max(stamp)
    stamp = torch.FloatTensor(stamp)

    return stamp, max_stamp, meta