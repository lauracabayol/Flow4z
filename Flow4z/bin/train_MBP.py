import torch
import yaml
import sys
import os

# Assuming necessary imports for SBP_model_multExp, NF_model_MBPz, etc.
sys.path.append('../SBP')
sys.path.append('../MBP')

sys.path.append('../bookkeeper')
sys.path.append('../data')
sys.path.append('../utils')

from SBP import predict_flux
from SBP_models import SBP_model_multExp
from MBP_models import NF_model_MBP
from dataloader import create_dataloaders

from MBPz import MBPz  # Replace with the actual module name

def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file = sys.argv[1]  # First argument is the path to the config file
    config = load_config(config_file)
    
    mbpz = MBPz(
        model_path_sbp=config['model_path_sbp'],
        data_dir_path=config['data_dir_path'],
        nepochs=config['nepochs'],
        lr=config['lr'],
        batch_size=config['batch_size'],
        nexp=config.get('nexp', 3),  # Use value from config or default to 3
        verbose=config.get('verbose', True),
        save_path=config['save_path'],
        file_type=config['file_type'],
        flow_type=config['flow_type'],
        predict_photoz=config['predict_photoz'],
    )
    
    mbpz._train_model()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_mbpz.py <config_file.yaml>")
        sys.exit(1)
    main()
