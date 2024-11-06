import sys
import yaml
import argparse
import torch
from loguru import logger
from pathlib import Path


sys.path.append('../SBP')
sys.path.append('../MBP')

from SBP import SBP
from MBPz import MBPz


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    # Load the config file
    print("Loading configuration file...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration file loaded successfully.")

    # Set up the device (GPU or CPU)
    print("Setting up the device...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config['photometry_type'] == 'SBP':
        script_name = 'SBP'
        model = SBP(sbp_version = config['sbp_version'],
                          zp_calib=config['zp_calib'],
                          zp_calib_err=config['zp_calib_err'],
                          nexp = config['nexp'],
                          bands = config['bands']
                         )
    elif config['photometry_type'] == 'MBP':
        script_name = 'MBP'
        model = MBPz(
            sbp_version=config['sbp_version'],
            mbp_version=config['mbp_version'],
            zp_calib_err=config['zp_calib_err'],
            nexp=config.get('nexp', 3),  
            save_path=config['output_model'],
            file_type=config['file_type'],
            predict_photoz=config['predict_photoz'],
        )
    else:
        raise ValueError(f"Unsupported photometry_type: {config['photometry_type']}. Please choose 'SBP' or 'MBP'.")

    # Train model
    print("Starting model training...")
    trained_model = model.train(
        data_dir=Path(config['data_dir']),
        training_hyperparams=config['hyperparams'],
        metadata_dir=Path(config['metadata_dir']))
    
    print("Model training completed.")

    # Prepare metadata including logs
    print("Preparing metadata including logs...")
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

    # Save the trained model with metadata
    print(f"Saving the trained model with metadata to {config['output_model']}...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'metadata': metadata
    }, config['output_model'])
    print("Model and metadata saved successfully.")

if __name__ == '__main__':
    main()
