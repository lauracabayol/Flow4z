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

    if test_size < len(dset):
        dset_train, dset_test = torch.utils.data.random_split(dset, [len(dset) - test_size, test_size])
        
        # Prefetch first few batches of training and validation data
        prefetch_size = min(batch_size * 2, len(dset_train))
        dset.prefetch_batch(list(range(prefetch_size)))
        dset.prefetch_batch(list(range(len(dset) - test_size, len(dset))))

        # Create DataLoaders with worker init function
        def worker_init_fn(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:  # single-process data loading
                return
            
            dataset = worker_info.dataset
            per_worker = len(dataset) // worker_info.num_workers
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker
            
            # Prefetch this worker's chunk
            if isinstance(dataset, torch.utils.data.Subset):
                original_dataset = dataset.dataset
                worker_indices = [dataset.indices[i] for i in range(start_idx, end_idx)]
                original_dataset.prefetch_batch(worker_indices)
            else:
                dataset.prefetch_batch(range(start_idx, end_idx))

        loader_val = DataLoader(dset_test, 
                              batch_size=batch_size, 
                              shuffle=False,
                              worker_init_fn=worker_init_fn,
                              num_workers=4)  # Adjust num_workers as needed
        
        loader_train = DataLoader(dset_train, 
                                batch_size=batch_size, 
                                shuffle=True,
                                worker_init_fn=worker_init_fn,
                                num_workers=4)  # Adjust num_workers as needed
        
        return loader_train, loader_val
    else:
        # For single loader case
        prefetch_size = min(batch_size * 2, len(dset))
        dset.prefetch_batch(list(range(prefetch_size)))
        
        loader = DataLoader(dset, 
                          batch_size=batch_size, 
                          shuffle=False,
                          worker_init_fn=worker_init_fn,
                          num_workers=4)  # Adjust num_workers as needed
        return loader
