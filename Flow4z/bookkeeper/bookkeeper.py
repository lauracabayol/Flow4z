import os
import time
import json

class Bookkeeper_training:
    def __init__(self, log_file):
        """
        Initializes the Bookkeeper_training class.

        Args:
            log_file (str): Path to the log file where training logs will be stored.
        """
        self.log_file = log_file
        self.logs = []

        # Load existing logs if the file exists
        self.load_log()

    def log_training(self,
                     script_name,
                     input_directory,
                     output_model,
                     training_hyperparams,
                     bands,
                     multiple_exposures,
                     zp_calib=None):
        """
        Logs training information.

        Args:
            script_name (str): Name of the script being run.
            input_directory (str): Directory where input data is stored.
            output_model (str): Path where the trained model will be saved.
            training_hyperparams (dict): Dictionary of training hyperparameters.
            bands (list): List of photometric bands used.
            multiple_exposures (bool): Whether multiple exposures were used.
            zp_calib (float or None): Zero-point calibration value (if any).
        """
        entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'script_name': script_name,
            'input_directory': input_directory,
            'output_file': output_model,
            'hyperparam_dict': training_hyperparams,
            'bands': bands,
            'multiple_exposures': multiple_exposures,
            'zp_calib': zp_calib
        }

        self.logs.append(entry)
        self._write_log()

    def _write_log(self):
        """
        Writes the current logs to the log file.
        If the file already exists, it appends new logs to the existing ones.
        """
        with open(self.log_file, 'a') as file:
            json.dump(self.logs, file, indent=4)

    def load_log(self):
        """
        Loads existing logs from the log file, if it exists.
        """
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as file:
                self.logs = json.load(file)
