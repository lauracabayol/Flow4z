import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List

def create_dataloaders(path_data: Path | str, 
                       bands: List[str], 
                       path_metadata: Path | str = None,
                       batch_size: int = 100, 
                       test_size: int = 500,
                       nexp: int = 3,
                       zp_calib: bool = None,
                       zp_calib_err: float = 0,
                       file_type='features'):
    """
    Create PyTorch DataLoader objects for training and validation from a dataset directory.

    Args:
        path_data (str): Path to the dataset directory.

    Returns:
        DataLoader: DataLoader for training set.
        DataLoader: DataLoader for validation set.
    """

    # Create a DataSet instance based on the provided dataset directory
    if str(path_data).endswith('.zarr'):
        from Flow4z.data.dataset_zarr import DataSet
        dset = DataSet(data_dir=path_data,
                   metadata_dir=path_metadata,
                   bands=bands,
                   nexp=nexp)
    else:
        from Flow4z.data.dataset import DataSet
        dset = DataSet(data_dir=path_data,
                    bands=bands,
                    multiple_exps=True,
                    nexp=nexp,
                    zp_calib=zp_calib,
                    zp_calib_err=zp_calib_err,
                    file_type=file_type)

    if test_size<len(dset):
        
        dset_train, dset_test = torch.utils.data.random_split(dset, [len(dset) - test_size, test_size])
        # Create DataLoader for the validation set
        loader_val = DataLoader(dset_test, batch_size=batch_size, shuffle=False)
        # Create DataLoader for the training set
        loader_train = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
        return loader_train, loader_val
    else:
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False)
        return loader
