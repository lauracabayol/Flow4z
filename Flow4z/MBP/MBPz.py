import torch
import numpy as np
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
import logging
from pathlib import Path
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

client = MlflowClient()

from Flow4z.utils.logging_config import setup_logging
from Flow4z.MBP.MBP_models import NF_model_MBPz
from Flow4z.data.dataloader import create_dataloaders
from Flow4z.SBP import SBP

logger = logging.getLogger(__name__)
setup_logging()

@dataclass
class MBPz:
    sbp_version: str
    restore: bool = False
    mbp_version: str = None
    bands: List[str] = ["CFHT_U", "CFHT_G", "CFHT_R", "CFHT_I", "CFHT_Z"]
    nexp: int = 3
    save_path: str = None
    file_type: str = "features"
    predict_photoz: bool = True
    zp_calib_err: int = 0
    ntransformation: int = 8
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"

    def __post_init__(self):
        """Initialize the MBPz model with optional loading of pretrained models.

        Sets up device, loads SBP model, initializes normalizing flow model, and configures
        parameters like number of exposures, transformations, etc.
        """
        logger.info("Initializing MBPz model...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Initialize SBP model
        logger.info("Loading SBP model...")
        model_name = "SBP"
        model_uri = f"models:/{model_name}/{self.sbp_version}"
        self.model_sbp = mlflow.pytorch.load_model(model_uri).to(self.device)

        self.nbands = len(self.bands)
        self.zp_calib_err = self.zp_calib_err

        # Initialize normalizing flow model
        logger.info("Initializing normalizing flow model...")

        if self.restore:
            if self.mbp_version is None:
                msg = "mbp_version must be provided when restore=True"
                logger.error(msg)
                raise ValueError(msg)

            logger.info("Loading MBP model...")
            model_name = "MBP"
            print(model_name, self.mbp_version)
            model_uri = f"models:/{model_name}/{self.mbp_version}"
            self.normflow = mlflow.pytorch.load_model(model_uri).to(self.device)

            # Access model parameters
            model_version_details = client.get_model_version("MBP", self.mbp_version)
            run_id = model_version_details.run_id
            run = client.get_run(run_id)
            params = run.data.params
            logger.info("Overwriting parameters to those of the loaded model...")
            self.predict_photoz = eval(params['predict_photoz'])
            self.batch_size = int(params['batch_size'])
            self.zp_calib_err = int(params['zp_calib_error'])
            self.input_dim = self.nbands + 1 if self.predict_photoz else self.nbands

        else:
            self.input_dim = self.nbands + 1 if self.predict_photoz else self.nbands
            self.normflow = NF_model_MBPz(dim_inputSpace=self.input_dim)

        self.normflow = self.normflow.to(self.device)

    def train(self, path_data: Path | str,
              path_metadata: Path | str,
              training_hyperparams: dict):
        """
        Trains the normalizing flow model using the provided data.

        Arguments:
        None (all necessary parameters are initialized in the __init__ method).
        """
        logger.info("Creating data loaders...")
        nbands = len(self.bands)

        loader_train, _ = create_dataloaders(
            path_data=path_data,
            path_metadata=path_metadata,
            bands=self.bands,
            nexp=self.nexp,
            batch_size=training_hyperparams["batch_size"],
            zp_calib_err=self.zp_calib_err,
            file_type=self.file_type,
        )

        optimizer = torch.optim.Adam(
            self.normflow.parameters(), lr=training_hyperparams["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        with mlflow.start_run() as run:
            logger.info(
                f"Starting training for {training_hyperparams['nepochs']} epochs..."
            )
            progress_bar = tqdm(range(training_hyperparams["nepochs"]), desc="Training epochs", unit="epoch")
            for epoch in progress_bar:
                epoch_loss = 0.0

                for meta, data, max_norm in loader_train:
                    z, lab = meta[:, 0, :, :], meta[:, 1, :, :]

                    lab = lab[:, :, 0] / max_norm.mean(2)

                    optimizer.zero_grad()

                    if self.file_type == "image":
                        raise NotImplementedError("Predicting features from images is not supported")
                    else:
                        features = data.view(len(data), -1)

                    input_nf = torch.cat((lab, meta[:, 0:1, 1]), dim=1) if self.predict_photoz else lab

                    z, log_jac_det = self.normflow(
                        input_nf.to(self.device), features.to(self.device)
                    )
                    loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
                    loss = loss.mean()

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    progress_bar.set_postfix({"loss": loss.item()})

                scheduler.step()

                # Log loss as a metric after each epoch
                mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)

            # Log and register the model with MLflow
            mlflow.pytorch.log_model(
                self.normflow, artifact_path="model", registered_model_name="MBP"
            )

            mlflow.set_tag("project", "Flow4z:MBP")
            mlflow.set_tag("input directory", f"{path_data}")
            mlflow.set_tag("Model_SBP", f"{self.sbp_version}")

            # Log parameters for the run
            mlflow.log_param("learning_rate", training_hyperparams["learning_rate"])
            mlflow.log_param("batch_size", training_hyperparams["batch_size"])
            mlflow.log_param("nepochs", training_hyperparams["nepochs"])
            mlflow.log_param("nexp", self.nexp)
            mlflow.log_param("zp_calib", True)
            mlflow.log_param("zp_calib_error", self.zp_calib_err)
            mlflow.log_param("predict_photoz", f"{self.predict_photoz}")

            logger.info("Training completed.")

        return self.normflow

    def process_catalog(self, data_dir: Path | str,
                        Nrealizations: int = 100,
                        return_distributions: bool = False):
        """
        Generates predictions for a given dataset, optionally returning features and distributions.

        Arguments:
        - path_data (str): Path to the directory containing the dataset for prediction.
        - nexp (int): Number of exposure samples to use in predictions. Default is 3.
        - Nrealizations (int): Number of realizations to generate per object. Default is 100.
        - return_features (bool): If True, return the features along with predictions. Default is False.
        - return_distributions (bool): If True, return the full distribution of predictions. Default is True.

        Returns:
        - preds_all (numpy.ndarray): Array containing all predictions if return_distributions is True.
        - photometric_preds_mean, photoz_preds_mean (numpy.ndarray): Mean predictions if only features are returned.
        """
        self.normflow = self.normflow.eval()
        batch_size = 1
        logger.info(f"Predicting dataset with {Nrealizations} realizations per object...")
        loader_test = create_dataloaders(
            path_data=data_dir,
            nexp=self.nexp,
            test_size=1000,
            batch_size=batch_size,
            bands=self.bands,
            zp_calib_err=self.zp_calib_err,
            file_type='features',
        )

        nobj = len(os.listdir(data_dir))
        preds_photometry_all = np.zeros(shape=(nobj, Nrealizations, self.nbands))
        photometry_true = np.zeros(shape=(nobj, self.nbands))

        preds_all_photoz = np.zeros((nobj, Nrealizations))
        photoz_true = np.zeros(shape=(nobj, 1))

        progress_bar = tqdm(loader_test, desc="Prediction Progress")

        for samp, (meta, features, max_norm) in enumerate(progress_bar):

            features = features.reshape(len(features), self.nbands * 10)
            condition = torch.tile(features, (Nrealizations, 1)).to(self.device)
            photometry_true[samp] = meta[:, :, 0]
            photoz_true[samp] = meta[:, 0, 1]

            z_test = torch.randn(Nrealizations, self.input_dim).to(self.device)
            preds, _ = self.normflow(z_test, condition, rev=True)

            preds = preds.detach().cpu().numpy()
            if self.predict_photoz:
                preds_photometry_all[samp] = preds[:, :-1] * max_norm.numpy()
                preds_all_photoz[samp] = preds[:, -1]
            else:
                preds_photometry_all[samp] = preds * max_norm.numpy()

        if return_distributions:
            return preds_photometry_all, preds_all_photoz

        # Calculate mean and std for photometric predictions
        photometric_preds_mean = np.nanmean(preds_photometry_all, axis=1)
        photometric_preds_err = np.nanstd(preds_photometry_all, axis=1)

        if self.predict_photoz:
            # Calculate mean and std for photoz predictions
            photoz_preds_mean = np.nanmean(preds_all_photoz, axis=1)
            photoz_preds_err = np.nanstd(preds_all_photoz, axis=1)
            return (photometric_preds_mean, photometric_preds_err, photometry_true,
                    photoz_preds_mean, photoz_preds_err, photoz_true)

        return photometric_preds_mean, photometric_preds_err, photometry_true
