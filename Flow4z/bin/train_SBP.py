import sys
import yaml
import argparse
import torch

sys.path.append('../SBP')
sys.path.append('../bookkeeper')
sys.path.append('../data')
sys.path.append('../utils')

from dataloader import create_dataloaders
from SBP import SBP
from SBP_models import SBP_model_multExp
from bookkeeper import Bookkeeper_training
from dataset import DataSet

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
    print(f"Device set to: {device}")

    sbp_trainer = SBP(model_path = =config['model_path'],
                      zp_calib=config['zp_calib'],
                      nexp = config['nexp'],
                      bands = config['bands']
                     )

    # Train model
    print("Starting model training...")
    trained_model = sbp_trainer.train(
        data_dir=config['data_dir'],
        training_hyperparams=config['hyperparams'])
    )
    print("Model training completed.")

    # Prepare metadata including logs
    print("Preparing metadata including logs...")
    metadata = {
        'model_metadata': {
            'script_name': 'SBP',
            'input_directory': config['data_dir'],
            'output_model': config['output_model'],
            'training_hyperparams': config['hyperparams'],
            'bands': config['bands'],
            'multiple_exposures': config['multiple_exposures'],
            'zp_calib': config['zp_calib']
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
