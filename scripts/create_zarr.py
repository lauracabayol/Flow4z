import zarr
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from loguru import logger

def create_zarr_dataset(data_dir: Path, zarr_output: Path) -> None:
    """
    Reads .npy files from multiple subdirectories and stores them in a single Zarr file.

    Parameters:
    - data_dir: str, the path to the directory containing the 10,000 subdirectories.
    - zarr_output: str, the path to the Zarr file to store the combined data.
    """
    logger.info(f'Creating Zarr dataset from {data_dir} to {zarr_output}')
    # Initialize Zarr storage (root group)
    zarr_store = zarr.open_group(zarr_output, mode='w')
    total_subdirs = len(os.listdir(data_dir.as_posix()))

    # Loop over each subdirectory with a progress bar
    for subdir_idx in tqdm(range(total_subdirs), desc="Processing Subdirectories", unit="subdir"):
        subdir = os.path.join(data_dir, f'data_{subdir_idx}')

        # Create a Zarr group for each subdirectory to store its .npy files
        group_name = f'data_{subdir_idx}'
        subdir_group = zarr_store.create_group(group_name)

        # Loop over the files in each subdirectory
        for band in range(455, 855, 10):
            for exp in [0, 1, 2]:
                file_name = f'cutout_pau_nb{band}_exp{exp}.npy'
                file_path = os.path.join(subdir, file_name)

                # Load the .npy file
                data = np.load(file_path)

                # Store each .npy file in its respective group within Zarr
                zarr_dataset_name = f'nb{band}_exp{exp}'
                subdir_group.create_dataset(zarr_dataset_name, data=data, chunks=True, overwrite=True)

    logger.info(f'Data successfully stored in {zarr_output}')

def create_zarr_metadata(data_dir: Path, zarr_output: Path) -> None:
    """
    Reads .npy files from multiple subdirectories and stores them in a single Zarr file.

    Parameters:
    - data_dir: str, the path to the directory containing the 10,000 subdirectories.
    - zarr_output: str, the path to the Zarr file to store the combined data.
    """
    logger.info(f'Creating Zarr metadata from {data_dir} to {zarr_output}')
    # Initialize Zarr storage (root group)
    zarr_store = zarr.open_group(zarr_output, mode='w')
    total_subdirs = len(os.listdir(data_dir.as_posix()))

    # Loop over each subdirectory with a progress bar
    for subdir_idx in tqdm(range(total_subdirs), desc="Processing Subdirectories", unit="subdir"):
        subdir = os.path.join(data_dir, f'data_{subdir_idx}')

        # Create a Zarr group for each subdirectory to store its metadata
        group_name = f'data_{subdir_idx}'
        subdir_group = zarr_store.create_group(group_name)

        # Loop over the files in each subdirectory
        for band in range(455, 855, 10):
            for exp in [0, 1, 2]:
                # Assuming the metadata filenames follow this format
                metadata_file_name = f'metadata_pau_nb{band}_exp{exp}.npy'
                metadata_file_path = os.path.join(subdir, metadata_file_name)

                # Load the metadata .npy file
                metadata = np.load(metadata_file_path)

                metadata_shape = (1,3)  # Define shape according to your needs
                metadata_dataset_name = f'metadata_nb{band}_exp{exp}'
                metadata_dataset = subdir_group.create_dataset(metadata_dataset_name, data=metadata, chunks=True, overwrite=True)

                # Store metadata as an array or object
                metadata_dataset[0] = metadata[0]  # Storing as an object, adjust as needed

                # Alternatively, add metadata as attributes if it's appropriate
                metadata_dataset.attrs.update({
                    'band': band,
                    'exposure': exp,
                })

    logger.info(f'Data successfully stored in {zarr_output}')



def parse_args():
    parser = argparse.ArgumentParser(description='Create a Zarr dataset from multiple .npy files')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the input data subdirectories')
    parser.add_argument('--zarr_output', type=str, required=True, help='Path to save the output Zarr file')
    parser.add_argument('--dataset', type=str, required=False, default='dataset', help='Type of dataset to create: "dataset" or "metadata"')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset=='dataset':
        create_zarr_dataset(Path(args.data_dir), Path(args.zarr_output))
    if args.dataset=='metadata':
        create_zarr_metadata(Path(args.data_dir), Path(args.zarr_output))