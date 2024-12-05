#! /usr/bin/env python
import numpy as np
import pandas as pd
import argparse
import sys
import os
import torch
import logging
from mlflow.tracking import MlflowClient
client = MlflowClient()


from Flow4z.utils.logging_config import setup_logging
logger = logging.getLogger(__name__)
setup_logging()

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

from Flow4z.SBP.SBP import SBP
from Flow4z.MBP.MBPz import MBPz
from Flow4z.utils.mlflow_ui import start_mlflow_ui


def main():
    """Main function to run the process_catalog from the command line."""
    # Parse command-line arguments
    client = MlflowClient()

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
        "--metadata_dir",
        type=str,
        required=False,
        help="Path to the directory with metadata to evaluate.",
    )
    parser.add_argument(
        "--sbp_version",
        type=int,
        required=False,
        help="MLFlow version of SBP trained model",
    )
    parser.add_argument(
        "--return_features",
        type=bool,
        required=False,
        default=False,
        help="Return features",
    )

    client = MlflowClient()

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Retrieve arguments
    mbp_version = args.mbp_version
    data_dir = args.data_dir
    metadata_dir = args.metadata_dir
    sbp_version = args.sbp_version
    return_features = args.return_features
    # Determine output file path
    dir_name = data_dir.rstrip("/").split("/")[-1]

    if mbp_version is None:
        photometry_type = "SBP"
    else:
        photometry_type = "MBP"

    if photometry_type == "SBP":
        logger.info(
            f"Running process_catalog with SBP model version: {sbp_version} "
            f"and data directory: {data_dir}"
        )
        model = SBP(
            restore=True,
            sbp_version=sbp_version,
            bands=[f"pau_nb{i}" for i in range(455, 855, 10)]
        )
        flux_predictions, flux_predictions_err, true_fluxes = model.process_catalog(
            data_dir,
            metadata_dir,
            return_features = return_features
        )

        output_file = (
            f"/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/"
            f"SBP_{dir_name}_{sbp_version}.csv"
        )
    elif photometry_type == "MBP":
        logger.info(
            f"Running process_catalog with MBP model version: {mbp_version} "
            f"and data directory: {data_dir}"
        )

        model_version_details = client.get_model_version('MBP', mbp_version)
        run_id = model_version_details.run_id
        run = client.get_run(run_id)
        params = run.data.params

        model = MBPz(
            sbp_version=sbp_version,
            mbp_version=mbp_version,
            predict_photoz=eval(params['predict_photoz']),
            bands=[f"pau_nb{i}" for i in range(455, 855, 10)],
            nexp=params['nexp'],
        )
        output_file = (
            f"/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/"
            f"MBP_{dir_name}_{mbp_version}.csv"
        )

        # Run model to process catalog and generate predictions
        if eval(params['predict_photoz']):
            flux_predictions, flux_predictions_err, true_fluxes, photoz, photoz_err, redsfhit = model.process_catalog(data_dir,
                                                                                                                     metadata_dir)                           
            # Log results and save to file
            catz = pd.DataFrame(
                np.c_[photoz.flatten(), photoz_err.flatten(), redsfhit.flatten()],
                columns=["photoz_pred", "photoz_err_pred", "redshift"],
            )
            output_file_z = (
                f"/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/"
                f"z_{dir_name}_{mbp_version}.csv"
            )
            catz.to_csv(output_file_z, header=True, sep=",")
        else:
            flux_predictions, flux_predictions_err, true_fluxes = model.process_catalog(data_dir,
                                                                                        metadata_dir)

    # Log results and save to file
    cat = pd.DataFrame(
        np.c_[
            flux_predictions.flatten(),
            flux_predictions_err.flatten(),
            true_fluxes.flatten()
        ],
        columns=["photometry_pred", "photometry_err_pred", "photometry_true"],
    )
    cat.to_csv(output_file, header=True, sep=",")
    logger.info(f"Catalog saved to {output_file} and logged to MLflow.")


if __name__ == "__main__":
    #process = start_mlflow_ui()
    main()
    #stop_mlflow_ui(process)
