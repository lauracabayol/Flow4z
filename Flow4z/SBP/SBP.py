from loguru import logger
import numpy as np
import sys
from scipy.stats import norm
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow
import mlflow.pytorch
from mlflow.exceptions import MlflowException
from loguru import logger

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Custom modules
sys.path.append("../data")
from dataset import DataSet
from load_image import load_image

sys.path.append("/.")
from SBP_models import SBP_model_multExp

sys.path.append("../utils")
from dataloader import create_dataloaders

from mlflow.tracking import MlflowClient
client = MlflowClient()



class SBP:
    def __init__(
        self, 
        sbp_version=None, 
        zp_calib=True, 
        zp_calib_err=0, 
        nexp=3, 
        bands=None
    ):
        """Initialize the SBP model with the option to load a pretrained model."""

        # Set the device for training (use GPU if available)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device set to: {self.device}")

        # Initialize the model architecture with a calibration parameter
        logger.info("Initializing the SBP model...")
        self.model = SBP_model_multExp(zp_calib=zp_calib).to(self.device)

        # If a model path is provided, load the pre-trained model
        if sbp_version is not None:
            logger.info(f"Loading MLFlow version: {sbp_version}")
            model_name = "SBP"
            model_uri = f"models:/{model_name}/{sbp_version}"
            model_version_details = client.get_model_version("SBP", sbp_version)

            run_id = model_version_details.run_id
            run = client.get_run(run_id)
            params = run.data.params   
            
            self.model = mlflow.pytorch.load_model(model_uri)
            self.bands=bands#['CFHT_U','CFHT_G','CFHT_R','CFHT_I','CFHT_Z']
            self.zp_calib = True
            self.nexp = params['nexp'],
            self.zp_calib_err=eval(params['zp_calib_err']),
            
        else:
            self.bands = bands
            self.zp_calib = zp_calib
            self.nexp = nexp
            self.zp_calib_err = zp_calib_err

    def train(self, data_dir, training_hyperparams):
        self.model = self.model.train()
        mlflow.autolog()

        nepochs = training_hyperparams["nepochs"]
        batch_size = training_hyperparams["batch_size"]
        lr = training_hyperparams["learning_rate"]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        nbands = len(self.bands)

        loader_train, loader_val = create_dataloaders(
            path_data=data_dir,
            bands=self.bands,
            batch_size=batch_size,
            zp_calib=self.zp_calib,
            nexp=self.nexp,
            test_size=2,
            file_type="image",
        )

        with mlflow.start_run():
            for epoch in range(nepochs):
                progress_bar = tqdm(
                    loader_train, desc=f"Epoch {epoch + 1}/{nepochs}", unit="epoch"
                )
                epoch_loss = 0

                for meta, stamp, max_norm in progress_bar:
                    optimizer.zero_grad()

                    meta = meta.squeeze(1)

                    z, lab = meta[:, :, :, 0], meta[:, :, :, 1]
                    lab = lab / max_norm
                    stamp = (
                        stamp.reshape(len(stamp) * nbands, stamp.shape[2], 60, 60)
                        .unsqueeze(1)
                        .float()
                    )
                    lab = lab[:, :, 0].reshape(len(lab) * nbands).unsqueeze(1)

                    if self.zp_calib:
                        zp = meta[:, :, :, 2]
                        zp = zp * torch.normal(
                            1, self.zp_calib_err / 100, size=zp.shape
                        )
                        zp = zp.reshape(len(zp) * nbands, self.nexp)

                        flux, logalpha, logsig, _ = self.model(
                            stamp.to(self.device), zp.to(self.device)
                        )
                    else:
                        flux, logalpha, logsig, _ = self.model(stamp.to(self.device))

                    logsig = torch.clamp(logsig, -6, 6)
                    sig = torch.exp(logsig)

                    log_prob = (
                        logalpha
                        - 0.5 * (flux - lab.to(self.device)).pow(2) / sig.pow(2)
                        - logsig
                    )
                    log_prob = torch.logsumexp(log_prob, 1)
                    loss = -log_prob.mean()

                    epoch_loss += loss

                    loss.backward()
                    optimizer.step()

                    progress_bar.set_postfix(loss=loss.item())

                    mlflow.log_metric("loss", loss.item(), step=epoch)

                scheduler.step()
                # Log loss as a metric after each epoch
                mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
                logger.info(
                    f"Epoch {epoch + 1}/{nepochs} completed. Median loss: {np.median(loss.detach().cpu().numpy())}"
                )

            # Log and register the model with MLflow
            mlflow.pytorch.log_model(
                self.model, artifact_path="model", registered_model_name="SBP"
            )

            mlflow.set_tag("project", "Flow4z:SBP")
            mlflow.set_tag("input directory", f"{data_dir}")
            mlflow.set_tag("zp_calib", f"{self.zp_calib}")
            mlflow.set_tag("zp_calib_err", f"{self.zp_calib_err}")

            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("nepochs", nepochs)
            mlflow.log_param("nexp", self.nexp)
            mlflow.log_param("zp_calib_err", f"{self.zp_calib_err}")

        return self.model

    def predict_flux(self, image, zp, return_pdf=False):
        self.model = self.model.eval()
        if self.zp_calib:
            flux, logalpha, logsig, features = self.model(image.to(self.device), zp)
        else:
            flux, logalpha, logsig, features = self.model(image.to(self.device))

        f = (torch.exp(logalpha) * flux).sum(1)
        ferr = (torch.exp(logalpha) * torch.exp(logsig)).sum(1)

        if return_pdf:
            alphas = torch.exp(logalpha).detach().cpu().numpy()
            sig = torch.exp(logsig).detach().cpu().numpy()
            alphas = alphas / alphas.sum(1)[:, None]
            flux = flux.detach().cpu().numpy()

            f_axis = np.concatenate(
                (np.arange(-10, 100, 0.1), np.arange(100, 500, 1)), 0
            )

            pdfs = alphas[:, :, None] * norm.pdf(
                f_axis, loc=flux[:, :, None], scale=sig[:, :, None]
            )
            pdfs = pdfs.sum(1)
            pdfs_norm = pdfs / pdfs.sum(1)[:, None]

            return f.detach().cpu().numpy(), features, pdfs_norm

        return f.detach().cpu().numpy(), ferr.detach().cpu().numpy(), features


    def process_catalog(self, data_dir, return_features=False):
        all_flux_predictions = []
        all_true_fluxes = []
        all_fluxerr_predictions = []
        zp = None

        nobj = len(os.listdir(data_dir))

        loader = create_dataloaders(
            data_dir,
            bands=self.bands,
            batch_size=1,
            zp_calib=self.zp_calib,
            file_type="image",
            test_size=nobj,
        )
        progress_bar = tqdm(loader, desc="Prediction Progress")

        logger.info("Loaders created")

        for i, (m, stamp, max_norm) in enumerate(progress_bar):

            # Reshape and move stamp to the appropriate device
            stamp = (
                stamp.reshape(len(stamp) * len(self.bands), 3, 60, 60)
                .unsqueeze(1)
                .float()
                .to(self.device)
            )

            # Reshape max_norm
            max_norm = max_norm.reshape(len(max_norm) * len(self.bands))

            # Apply zero-point calibration if required
            if self.zp_calib:

                zp = m[:, :, :, 2]
                zp = zp * torch.normal(1, self.zp_calib_err[0] / 100, size=zp.shape)
                zp = zp.reshape(len(zp) * len(self.bands), 3).to(self.device)

            # Predict flux for the current batch
            flux_pred, fluxerr_pred, features = self.predict_flux(stamp, zp)

            if return_features:
                logger.info("Saveing features")
                features_path = f"{data_dir}/data_{i}/features_{i}_{self.zp_calib_err[0]}.npy"
                np.save(features_path, features.detach().cpu().numpy())

            # Denormalize flux predictions if necessary
            flux_pred = flux_pred * max_norm.numpy()
            fluxerr_pred = fluxerr_pred * max_norm.numpy()

            # Store the predictions and true values
            all_flux_predictions.append(flux_pred)
            all_fluxerr_predictions.append(fluxerr_pred)
            
            ftrue = (
                m[:, :, 0, 1].reshape(len(m) * len(self.bands)).detach().cpu().numpy()
            )
            all_true_fluxes.append(ftrue)
        # Convert lists to arrays and log final sizes
        all_flux_predictions = np.concatenate(all_flux_predictions)
        all_fluxerr_predictions = np.concatenate(all_fluxerr_predictions)
        all_true_fluxes = np.concatenate(all_true_fluxes)
        logger.info(
            "All flux predictions and true fluxes concatenated. Total size: {}",
            len(all_flux_predictions),
        )

        return all_flux_predictions, all_fluxerr_predictions, all_true_fluxes
