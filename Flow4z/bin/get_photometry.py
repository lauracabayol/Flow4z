cleimport numpy as np
import pandas as pd
import argparse
import sys
import os
import mlflow
import mlflow.pytorch
import torch
from loguru import logger
from mlflow.tracking import MlflowClient

# Set up the Loguru logger
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Set MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
client = MlflowClient()

# Import SBP and MBPz modules
sys.path.append("../SBP")
sys.path.append("../MBP")
from SBP import SBP
from MBPz import MBPz


def main():
    """Main function to run the process_catalog from the command line."""

    client = MlflowClient()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a neural network on a dataset."
    )
    parser.add_argument(
        "--mbp_version",
        type=int,
        required=False,
        help="MLFlow version of MBP trained model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory with data to evaluate.",
    )
    parser.add_argument(
        "--sbp_version",
        type=int,
        required=False,
        help="MLFlow version of SBP trained model",
    )
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Retrieve arguments
    mbp_version = args.mbp_version
    data_dir = args.data_dir
    sbp_version = args.sbp_version

    # Determine output file path
    dir_name = data_dir.rstrip("/").split("/")[-1]

    if mbp_version is None:
        photometry_type = "SBP"
    else:
        photometry_type = "MBP"

    # Start an MLflow run
    with mlflow.start_run():
        # Log input arguments
        mlflow.log_param("sbp_version", sbp_version)
        mlflow.log_param("mbp_version", mbp_version)
        mlflow.log_param("data_dir", data_dir)

        if photometry_type == "SBP":
            logger.info(
                f"Running process_catalog with SBP model version: {sbp_version} and data directory: {data_dir}"
            )
            model = SBP(sbp_version)
            flux_predictions, flux_predictions_err, true_fluxes = model.process_catalog(data_dir)

            
            output_file = f"/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/SBP_{dir_name}_{sbp_version}.csv"
        elif photometry_type == "MBP":
            logger.info(
                f"Running process_catalog with MBP model version: {mbp_version} and data directory: {data_dir}"
            )

            model_version_details = client.get_model_version('MBP', mbp_version)
            run_id = model_version_details.run_id
            run = client.get_run(run_id)
            params = run.data.params
            
            model = MBPz(mbp_version=mbp_version,
                        predict_photoz=eval(params['predict_photoz']),
                        nexp=params['nexp'],
                         
                        )
            output_file = f"/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/MBP_{dir_name}_{mbp_version}.csv"

            # Run model to process catalog and generate predictions
            if eval(params['predict_photoz'])==True:
                flux_predictions, flux_predictions_err, true_fluxes, photoz, photoz_err, redsfhit = model.process_catalog(data_dir)   
                # Log results and save to file
                catz = pd.DataFrame(
                    np.c_[photoz.flatten(), photoz_err.flatten(), redsfhit.flatten()],
                    columns=["photoz_pred", "photoz_err_pred", "redshift"],
                )
                output_file_z = f"/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/z_{dir_name}_{mbp_version}.csv"
                catz.to_csv(output_file_z, header=True, sep=",")
            else:
                flux_predictions, flux_predictions_err, true_fluxes = model.process_catalog(data_dir)

        
        # Log results and save to file
        cat = pd.DataFrame(
            np.c_[flux_predictions.flatten(), flux_predictions_err.flatten(), true_fluxes.flatten()],
            columns=["photometry_pred", "photometry_err_pred", "photometry_true"],
        )
        cat.to_csv(output_file, header=True, sep=",")
        logger.info(f"Catalog saved to {output_file} and logged to MLflow.")


if __name__ == "__main__":
    main()
