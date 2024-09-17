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
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_path> <data_dir>")
        sys.exit(1)
    
    # Parse command-line arguments
    model_path = sys.argv[1]
    data_dir = sys.argv[2]

    print(f"Running process_catalog with model: {model_path} and data directory: {data_dir}")

    # Start an MLflow run
    with mlflow.start_run():
        # Log input arguments
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("data_dir", data_dir)

        # Initialize SBP object and call process_catalog
        sbp = SBP(model_path=model_path)
        flux_predictions, true_fluxes = sbp.process_catalog(data_dir)

        print(flux_predictions.shape, true_fluxes.shape)

        # Log the evaluation metrics
        mlflow.log_metric("num_flux_predictions", len(flux_predictions))

        # Create a DataFrame to store the results
        cat = pd.DataFrame(np.c_[flux_predictions, true_fluxes], 
                           columns=['pred', 'true'])

        # Determine output file path
        dir_name = data_dir.rstrip('/').split('/')[-1]
        model_name = model_path.rstrip('/').split('/')[-1]
        output_file = f'/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/SBP_{dir_name}_{model_name}_evaluation.csv'

        # Save the catalog to a CSV file
        cat.to_csv(output_file, header=True, sep=',')

        # Log the output file
        mlflow.log_artifact(output_file)

        # End of the run
        print(f"Catalog saved to {output_file} and logged to MLflow.")

if __name__ == "__main__":
    main()
