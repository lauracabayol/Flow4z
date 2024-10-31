#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import sys
import os
import mlflow
import mlflow.pytorch
sys.path.append('../SBP')
from SBP import SBP

def main():
    """Main function to run the process_catalog from the command line."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run feature extraction using a trained model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved neural network model.")
    args = parser.parse_args()
    print(f"Running process_catalog with model: {args.model_path} and data directory: {args.data_dir}")

    # Start an MLflow run
    with mlflow.start_run():
        # Log input arguments
        mlflow.log_param("model_path", args.model_path)
        mlflow.log_param("data_dir", args.data_dir)

        # Initialize SBP object and call process_catalog
        sbp = SBP(model_path=args.model_path)
        sbp.get_features(args.data_dir)

if __name__ == "__main__":
    main()






