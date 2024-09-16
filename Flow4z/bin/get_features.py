#!/usr/bin/env python

import argparse
import torch
import os
import sys

sys.path.append('../SBP')
from SBP import get_features
from SBP_models import SBP_model_multExp

sys.path.append('../data')
from load_image import load_image

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run feature extraction using a trained model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved neural network model.")
    parser.add_argument("--nexp", type=int, default=3, help="Number of exposures (default: 3).")

    args = parser.parse_args()

    # Inform the user that the process is starting
    print(f"Starting feature extraction with the following settings:")
    print(f" - Dataset directory: {args.data_dir}")
    print(f" - Model path: {args.model_path}")
    print(f" - Number of exposures: {args.nexp}")

    # Run the feature extraction
    try:
        get_features(
            data_dir=args.data_dir,
            model_path=args.model_path,
            nexp=args.nexp,
        )
        print("Feature extraction completed successfully.")
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")

if __name__ == "__main__":
    main()
