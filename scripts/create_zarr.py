import zarr
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from loguru import logger

def create_zarr_dataset(data_dir: Path, zarr_output: Path) -> None:
    """
    Reads .npy files and stores them in a single Zarr file, with each subdirectory
    containing an array of shape (40, 3, 60, 60) representing all images.
    """
    logger.info(f'Creating Zarr dataset from {data_dir} to {zarr_output}')
    zarr_store = zarr.open_group(zarr_output, mode='a')
    total_subdirs = len(os.listdir(data_dir.as_posix()))

    # Calculate dimensions
    n_bands = len(range(455, 855, 10))  # 40 bands
    n_exposures = 3
    img_size = 60  # Assuming 60x60 images

    #for subdir_idx in tqdm(range(total_subdirs), desc="Processing Subdirectories", unit="subdir"):
    for subdir_idx in tqdm(range(5024,5025), desc="Processing Subdirectories", unit="subdir"):
        subdir = data_dir / f'data_{subdir_idx}'
        
        # Pre-allocate array for all images in this subdirectory
        all_images = np.zeros((n_bands, n_exposures, img_size, img_size))
        
        # Fill the array
        for i, band in enumerate(range(455, 855, 10)):
            for j, exp in enumerate([0, 1, 2]):
                file_name = f'cutout_pau_nb{band}_exp{exp}.npy'
                file_path = subdir / file_name
                data = np.load(file_path)
                all_images[i, j] = data

        # Store as a single dataset with appropriate chunking
        chunk_size = (10, 3, 60, 60)  # Adjust these values as needed
        zarr_store.create_dataset(f'data_{subdir_idx}', 
                                data=all_images,
                                chunks=chunk_size,
                                overwrite=True)

    logger.info(f'Data successfully stored in {zarr_output}')

def create_zarr_metadata(data_dir: Path, zarr_output: Path) -> None:
    """
    Creates a Zarr store where each subdirectory's metadata is stored as a single array
    of shape (40, 3, 3) representing metadata for all bands and exposures.
    """
    logger.info(f'Creating Zarr metadata from {data_dir} to {zarr_output}')
    zarr_store = zarr.open_group(zarr_output, mode='a')
    total_subdirs = len(os.listdir(data_dir.as_posix()))

    # Calculate dimensions
    n_bands = len(range(455, 855, 10))  # 40 bands
    n_exposures = 3
    metadata_features = 3  # Assuming metadata has 3 features

    #for subdir_idx in tqdm(range(total_subdirs), desc="Processing Subdirectories", unit="subdir"):
    for subdir_idx in tqdm(range(9000,10000), desc="Processing Subdirectories", unit="subdir"):
        subdir = data_dir / f'data_{subdir_idx}'
        
        # Pre-allocate array for all metadata in this subdirectory
        all_metadata = np.zeros((n_bands, n_exposures, metadata_features))
        
        # Fill the array
        for i, band in enumerate(range(455, 855, 10)):
            for j, exp in enumerate([0, 1, 2]):
                metadata_file_name = f'metadata_pau_nb{band}_exp{exp}.npy'
                metadata_file_path = subdir / metadata_file_name
                metadata = np.load(metadata_file_path)
                all_metadata[i, j] = metadata[0]  # Assuming metadata[0] contains the values

        # Store as a single dataset with appropriate chunking
        chunk_size = (10, 3, 3)  # Adjust these values as needed
        dataset = zarr_store.create_dataset(f'data_{subdir_idx}', 
                                          data=all_metadata,
                                          chunks=chunk_size,
                                          overwrite=True)
        
        # Add attributes if needed
        dataset.attrs['bands'] = list(range(455, 855, 10))
        dataset.attrs['exposures'] = [0, 1, 2]

    logger.info(f'Metadata successfully stored in {zarr_output}')



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