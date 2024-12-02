import sys
import os
import yaml
import argparse
import torch
import logging
from pathlib import Path
from mlflow.tracking import MlflowClient
client = MlflowClient()

from Flow4z.utils.logging_config import setup_logging
from Flow4z.utils.mlflow_ui import start_mlflow_ui

logger = logging.getLogger(__name__)
setup_logging()
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

from Flow4z.SBP import SBP
from Flow4z.MBP import MBPz

def main():
    
    client = MlflowClient()
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    # Load the config file
    logger.info("Loading configuration file...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration file loaded successfully.")

    # Set up the device (GPU or CPU)
    logger.info("Setting up the device...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config['photometry_type'] == 'SBP':
        script_name = 'SBP'
        model = SBP.SBP(sbp_version = config['sbp_version'],
                          zp_calib=config['zp_calib'],
                          zp_calib_err=config['zp_calib_err'],
                          nexp = config['nexp'],
                          bands = config['bands']
                         )
    elif config['photometry_type'] == 'MBP':
        script_name = 'MBP'
        model = MBPz.MBPz(
            sbp_version=config['sbp_version'],
            mbp_version=config['mbp_version'],
            bands=config['bands'],
            restore=config['restore'],
            zp_calib=config['zp_calib'],
            nexp=config.get('nexp', 3),  
            file_type=config['file_type'],
            predict_photoz=config['predict_photoz'],
        )
    else:
        raise ValueError(f"Unsupported photometry_type: {config['photometry_type']}. Please choose 'SBP' or 'MBP'.")

    # Train model
    logger.info("Starting model training...")
    trained_model = model.train(
        path_data=Path(config['data_dir']),
        path_metadata=Path(config['metadata_dir']), 
        training_hyperparams=config['hyperparams'],
    )
    
    logger.info("Model training completed.")

    # Prepare metadata including logs
    logger.info("Preparing metadata including logs...")
    metadata = {
        'model_metadata': {
            'script_name': script_name,
            'input_directory': config['data_dir'],
            'output_model': config['output_model'],
            'training_hyperparams': config['hyperparams'],
            'bands': config['bands'],
            'multiple_exposures': config['multiple_exposures'],
            'nexp': config.get('nexp', 3),
            'zp_calib': config['zp_calib'],
            'sbp_version':config['sbp_version'],
            'predict_photoz':config['predict_photoz']
        }
    }

    # Save the trained model with metadata if output path exists
    if 'output_model' not in config or not config['output_model']:
        logger.warning("No output model path specified in config. Skipping model save.")
    else:
        logger.info(f"Saving the trained model with metadata to {config['output_model']}...")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'metadata': metadata
        }, config['output_model'])
        logger.info("Model and metadata saved successfully.")

if __name__ == '__main__':
    #start_mlflow_ui()
    main()