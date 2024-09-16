import sys
sys.path.append('../data')
from dataset import DataSet
import numpy as np
import pandas as pd


#torch modules
import torch
from torch.utils.data import DataLoader

def create_dataloaders(path_data, 
                       bands, 
                       batch_size=100, 
                       test_size=500,
                       nexp=3,
                       zp_calib=None,
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
    print(file_type)
    dset = DataSet(data_dir=path_data,
                   bands=bands,
                   multiple_exps=True,
                   nexp=nexp,
                   zp_calib=zp_calib,
                   file_type=file_type)

    # Split the dataset into training and test sets
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