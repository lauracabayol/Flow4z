import zarr
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def add_maxnorm(zarr_file, metadata_dir, metadata_dir_new, num_samples=10000):
    logger.info(f"Opening zarr stores: {zarr_file}, {metadata_dir}, {metadata_dir_new}")
    zarr_metadata_store = zarr.open_group(metadata_dir, mode="r")
    zarr_store = zarr.open_group(zarr_file, mode="r")
    new_met = zarr.open_group(metadata_dir_new, mode='a')

    num_samples = len(zarr_metadata_store)
    logger.info(f"Processing {num_samples} samples")
    for i in tqdm(range(0, num_samples), desc="Processing samples"):
        met = zarr_metadata_store[f"data_{i}"][:]
        stamp = zarr_store[f"data_{i}"][:]
        stamp = np.nan_to_num(stamp)
        max_stamp = np.max(stamp, axis=(2,3))
        new_met_data = np.c_[met, max_stamp[:,:,None]]
        new_met.create_dataset(f"data_{i}", data=new_met_data)
    logger.info("Finished processing all samples")

def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting add_maxnorm script")
    
    parser = argparse.ArgumentParser(description="Add max norm to metadata")
    parser.add_argument("--zarr_file", type=str, required=True, help="Path to zarr file")
    parser.add_argument("--metadata_dir", type=str, required=True, help="Path to metadata directory")
    parser.add_argument("--metadata_dir_new", type=str, required=True, help="Path to new metadata directory")
    
    args = parser.parse_args()
    logger.info(f"Arguments: zarr_file={args.zarr_file}, metadata_dir={args.metadata_dir}, metadata_dir_new={args.metadata_dir_new}")
    
    add_maxnorm(args.zarr_file, args.metadata_dir, args.metadata_dir_new)
    logger.info("Script completed successfully")

if __name__ == "__main__":
    main()
