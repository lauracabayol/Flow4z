import os
import torch
import numpy as np
from scipy.stats import norm

from tqdm import tqdm
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import zarr


import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Custom modules
from Flow4z.SBP.SBP_models import SBP_model_multExp
from Flow4z.data.dataloader import create_dataloaders
from Flow4z.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

@dataclass
class SBP:
    restore: bool = False
    sbp_version: Optional[str] = None
    zp_calib: bool = True
    zp_calib_err: float = 0
    nexp: int = 3
    bands: Optional[List[str]] = None
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"

    def __post_init__(self):
        """Initialize the SBP model with the option to load a pretrained model."""

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Set the device for training (use GPU if available)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device set to: {self.device}")

        # Initialize the model architecture with a calibration parameter
        logger.info("Initializing the SBP model...")
        self.model = SBP_model_multExp(zp_calib=self.zp_calib).to(self.device)

        if self.restore:
            if self.sbp_version is None:
                msg = "sbp_version must be provided when restore=True"
                logger.error(msg)
                raise ValueError(msg)
            logger.info(f"Loading MLFlow version: {self.sbp_version}")
            model_name = "SBP"
            model_uri = f"models:/{model_name}/{self.sbp_version}"
            model_version_details = client.get_model_version("SBP", self.sbp_version)

            run_id = model_version_details.run_id
            run = client.get_run(run_id)
            self.params = run.data.params   
            
            self.model = mlflow.pytorch.load_model(model_uri)
            self.nexp = int(self.params['nexp'])
            self.zp_calib_err=int(self.params['zp_calib_err'])
            

    def _process_batch(self, meta: torch.Tensor, 
                       stamp: torch.Tensor, 
                       max_norm: torch.Tensor, 
                       nbands: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Process a single batch of data."""
        z, lab = meta[:, 0, :, :], meta[:, 1, :, :]
        lab = lab[:, :, 0].reshape(len(lab) , nbands)
        lab = lab / max_norm
        
        # Reshape stamp and lab
        stamp = (
            stamp.reshape(len(stamp) * nbands, stamp.shape[2], 60, 60)
            .unsqueeze(1)
            .float()
        )
        lab = lab.reshape(len(lab) * nbands).unsqueeze(1)
        
        # Handle zero-point calibration
        #zp = None
        if self.zp_calib:
            zp = meta[:, 2, :, :]
            zp = zp * torch.normal(1, self.zp_calib_err / 100, size=zp.shape)
            zp = zp.reshape(len(zp) * nbands, self.nexp).to(self.device)
        
        return stamp, lab, zp

    def _compute_loss(self, stamp: torch.Tensor, 
                       lab: torch.Tensor, 
                       zp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute model loss for given inputs."""
        if zp is not None:
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
        return -log_prob.mean()

    def _log_training_metrics(self, epoch: int, 
                              nepochs: int, 
                              epoch_loss: torch.Tensor, 
                              loss: torch.Tensor) -> None:
        """Log training metrics to MLflow."""
        mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
        logger.info(
            f"Epoch {epoch + 1}/{nepochs} completed. "
            f"Median loss: {np.median(loss.detach().cpu().numpy())}"
        )

    def _log_final_metrics(self, data_dir: Path | str, 
                            training_hyperparams: dict) -> None:
        """Log final model metrics and parameters."""
        mlflow.pytorch.log_model(
            self.model, artifact_path="model", registered_model_name="SBP"
        )

        # Log tags
        tags = {
            "project": "Flow4z:SBP",
            "input directory": str(data_dir),
            "zp_calib": str(self.zp_calib),
            "zp_calib_err": str(self.zp_calib_err)
        }
        for key, value in tags.items():
            mlflow.set_tag(key, value)

        # Log parameters
        params = {
            "learning_rate": training_hyperparams["learning_rate"],
            "batch_size": training_hyperparams["batch_size"],
            "nepochs": training_hyperparams["nepochs"],
            "nexp": self.nexp,
            "zp_calib_err": str(self.zp_calib_err)
        }
        for key, value in params.items():
            mlflow.log_param(key, value)

    def train(self, data_dir: Path | str, 
              training_hyperparams: dict, 
              metadata_dir: Optional[Path | str] = None) -> None:
        """Main training loop."""
        self.model = self.model.train()
        mlflow.autolog()
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=training_hyperparams["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        nbands = len(self.bands)

        loader_train, loader_val = create_dataloaders(
            path_data=data_dir,
            path_metadata=metadata_dir,
            bands=self.bands,
            batch_size=training_hyperparams["batch_size"],
            zp_calib=self.zp_calib,
            nexp=self.nexp,
            test_size=2,
            file_type="image",
        )

        with mlflow.start_run():
            progress_bar = tqdm(range(training_hyperparams["nepochs"]), desc="Training epochs", unit="epoch")
            for epoch in progress_bar:
                epoch_loss = 0
                for meta, stamp, max_norm in loader_train:
                    optimizer.zero_grad()
                    
                    # Process batch and compute loss
                    stamp, lab, zp = self._process_batch(meta, stamp, max_norm, nbands)
                    loss = self._compute_loss(stamp, lab, zp)
                    
                    # Update model
                    epoch_loss += loss
                    loss.backward()
                    optimizer.step()

                    progress_bar.set_postfix(loss=loss.item())
                    mlflow.log_metric("loss", loss.item(), step=epoch)

                scheduler.step()
                self._log_training_metrics(epoch, training_hyperparams["nepochs"], epoch_loss, loss)

            # Log and register the model with MLflow
            mlflow.pytorch.log_model(
                self.model, artifact_path="model", registered_model_name="SBP"
            )

            mlflow.set_tag("project", "Flow4z:SBP")
            mlflow.set_tag("input directory", f"{data_dir}")
            mlflow.set_tag("zp_calib", f"{self.zp_calib}")
            mlflow.set_tag("zp_calib_err", f"{self.zp_calib_err}")

            mlflow.log_param("learning_rate", training_hyperparams["learning_rate"])
            mlflow.log_param("batch_size", training_hyperparams["batch_size"])
            mlflow.log_param("nepochs", training_hyperparams["nepochs"])
            mlflow.log_param("nexp", self.nexp)
            mlflow.log_param("zp_calib_err", f"{self.zp_calib_err}")

        return self.model

    def predict_flux(self, image: torch.Tensor, 
                      zp: Optional[torch.Tensor] = None, 
                      return_pdf: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict flux for a given image."""
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


    def process_catalog(self, data_dir: Path | str, 
                        metadata_dir: Path | str,
                        return_features: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a catalog of images and return flux predictions."""
        all_flux_predictions = []
        all_true_fluxes = []
        all_fluxerr_predictions = []

        batch_size = int(self.params["batch_size"])

        nobj = len(os.listdir(data_dir))
        loader = create_dataloaders(
            path_data=data_dir,
            path_metadata=metadata_dir,
            bands=self.bands,
            batch_size=batch_size,
            zp_calib=self.zp_calib,
            nexp=self.nexp,
            test_size=nobj,
            file_type="image",
        )
        progress_bar = tqdm(loader, desc="Prediction Progress")
        

        logger.info("Loaders created")

        for i, (meta, stamp, max_norm) in enumerate(progress_bar):
            nbands = len(self.bands)
            
            # Process the batch using _process_batch method
            stamp, lab, zp = self._process_batch(meta, stamp, max_norm, nbands)

            flux_pred, fluxerr_pred, features = self.predict_flux(stamp, zp)

            if return_features:
                logger.info("Saving features to zarr file")
                features = features.reshape((batch_size, nbands, 10))
                features_path = f"{data_dir}/features_v2.zarr"
                if i == 0:  # Create zarr file on first batch
                    self.features_store = zarr.open(features_path, mode='a')                
                
                # Store each sample's features in sequential groups
                for j, k in zip(range(batch_size), range(batch_size*i, batch_size*(i+1))):
                    group_name = f"data_{k}"
                    self.features_store.create_dataset(group_name, data=features[j].detach().cpu().numpy())

            # Denormalize flux predictions
            flux_pred = flux_pred * max_norm.flatten().numpy()
            fluxerr_pred = fluxerr_pred * max_norm.flatten().numpy()

            # Store the predictions and true values
            all_flux_predictions.append(flux_pred)
            all_fluxerr_predictions.append(fluxerr_pred)
            all_true_fluxes.append(lab.flatten().numpy())

        # Convert lists to arrays and log final sizes
        all_flux_predictions = np.concatenate(all_flux_predictions)
        all_fluxerr_predictions = np.concatenate(all_fluxerr_predictions)
        all_true_fluxes = np.concatenate(all_true_fluxes)
        logger.info(
            "All flux predictions and true fluxes concatenated. Total size: {}",
            len(all_flux_predictions),
        )

        return all_flux_predictions, all_fluxerr_predictions, all_true_fluxes
