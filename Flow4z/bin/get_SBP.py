import numpy as np
import pandas as pd
import argparse
import sys
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

    # Initialize SBP object and call process_catalog
    sbp = SBP(model_path=model_path)
    flux_predictions, true_fluxes = sbp.process_catalog(data_dir)

    cat = pd.DataFrame(np.c_[flux_predictions, true_fluxes], ['pred','true'])

    dir_name = data_dir.rstrip('/').split('/')[-1]
    model_name = model_path.rstrip('/').split('/')[-1]

    cat.to_csv(f'/nfs/pic.es/user/l/lcabayol/AI/Flow4z/catalogs/SBP_{dir_name}_{model_name}_{zp_calib}.csv', 
                   header =True,
                   sep =',')

if __name__ == "__main__":
    main()