import torch
from torch import nn, optim
import numpy as np
import datetime
import time
import os
import sys
from tqdm import tqdm  # Progress bar
import mlflow
import mlflow.pytorch
from mlflow.exceptions import MlflowException
from loguru import logger
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()


# Assuming necessary imports for SBP_model_multExp, NF_model_MBP, etc.
sys.path.append("../SBP")
sys.path.append("../MBP")
sys.path.append("../bookkeeper")
sys.path.append("../data")
sys.path.append("../utils")

from SBP_models import SBP_model_multExp
from MBP_models import NF_model_MBP
from dataloader import create_dataloaders
from SBP import SBP
from dataset import DataSet
from load_image import load_image


class MBPz:
    def __init__(
        self,
        sbp_version=None,
        mbp_version=None,
        nexp=3,
        save_path=None,
        file_type="features",
        predict_photoz=True,
        zp_calib_err=0,
        ntransformation=8,
    ):
        """
        Initializes the MBPz model, setting up paths, models, and hyperparameters.

        Arguments:
        - model_path_sbp (str): Path to the pre-trained SBP model checkpoint.
        - nexp (int): Number of exposure samples to generate. Default is 3.
        - save_path (str or None): Directory to save the trained model checkpoints. Default is None.
        - file_type (str): Type of input files (e.g., 'image', 'csv'). Default is 'image'.
        - predict_photoz (bool): Whether to predict photometric redshifts. Default is True.
        - ntransformation (int): Number of transformations in the normalizing flow. Default is 8.
        """
        print("Initializing MBPz model...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.file_type = file_type
        self.nexp = nexp
        self.ntransformation = ntransformation
        self.predict_photoz = predict_photoz
        self.save_path = save_path
        self.sbp_version = sbp_version

        
        client = MlflowClient()

        # Initialize SBP model
        logger.info("Loading SBP model...")
        model_name = "SBP"
        model_uri = f"models:/{model_name}/{sbp_version}"
        model = mlflow.pytorch.load_model(model_uri).to(self.device)

        # Retrieve zp_calib from model metadata
        zp_calib = True  # checkpoint['metadata']['model_metadata']['zp_calib']
        bands = [
            "CFHT_U",
            "CFHT_G",
            "CFHT_R",
            "CFHT_I",
            "CFHT_Z",
        ]  # checkpoint['metadata']['model_metadata']['bands']

        self.bands = bands
        self.nbands = len(bands)
        self.zp_calib_err=zp_calib_err

        
        # Initialize normalizing flow model
        logger.info("Initializing normalizing flow model...")

        if mbp_version is not None:
            logger.info("Loading MBP model...")
            model_name = "MBP"
            model_uri = f"models:/{model_name}/{mbp_version}"
            self.normflow  = mlflow.pytorch.load_model(model_uri).to(self.device)

            #access model parameters
            model_version_details = client.get_model_version(model_name, mbp_version)
            run_id = model_version_details.run_id
            run = client.get_run(run_id)
            params = run.data.params  
            logger.info("Overwritting parameters to those of the loaded model...")
            self.predict_photoz=eval(params['predict_photoz'])
            self.batch_size=int(params['batch_size'])
            self.zp_calib_err=int(params['zp_calib_error'])
            self.input_dim = self.nbands + 1 if self.predict_photoz else self.nbands

            
        else:
            self.input_dim = self.nbands + 1 if self.predict_photoz else self.nbands
            self.normflow = NF_model_MBP(dim_inputSpace=self.input_dim)
                    
        self.normflow = self.normflow.to(self.device)

    def train(self, data_dir, training_hyperparams):
        """
        Trains the normalizing flow model using the provided data.
    
        Arguments:
        None (all necessary parameters are initialized in the __init__ method).
        """
        logger.info("Creating data loaders...")
    
        loader_train, loader_val = create_dataloaders(
            data_dir,
            self.bands,
            nexp=self.nexp,
            batch_size=training_hyperparams["batch_size"],
            zp_calib_err=self.zp_calib_err,
            file_type=self.file_type,
        )
    
        optimizer = optim.Adam(
            self.normflow.parameters(), lr=training_hyperparams["learning_rate"]
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    
        with mlflow.start_run() as run:
            logger.info(
                f"Starting training for {training_hyperparams['nepochs']} epochs..."
            )
            for epoch in range(training_hyperparams["nepochs"]):
                epoch_loss = 0.0
                progress_bar = tqdm(
                    loader_train,
                    desc=f"Epoch {epoch + 1}/{training_hyperparams['nepochs']}",
                    unit="batch",
                )
                for meta, data, max_norm in progress_bar:
                    optimizer.zero_grad()
    
                    if self.file_type == "image":
                        features = torch.zeros(size=(len(data), self.nbands, 10))
                        flab = meta[:, :, 0, 1] / max_norm[:, :, 0]
                        for b in range(self.nbands):
                            f, feature = predict_flux(
                                self.model_sbp, data[:, b, :, :].unsqueeze(1)
                            )
                            features[:, b, :] = feature.detach()
                        features = features.reshape(len(features), self.nbands * 10)
    
                    elif self.file_type == "features":
                        flab = meta[:, :, 0] / max_norm
                        features = data.reshape(len(data), self.nbands * 10)
    
                    if self.predict_photoz:
                        input_nf = torch.cat((flab, meta[:, 0:1, 1]), dim=1)
                    else:
                        input_nf = flab
    
                    z, log_jac_det = self.normflow(
                        input_nf.to(self.device), features.to(self.device)
                    )
                    loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
                    loss = loss.mean()
    
                    loss.backward()
                    optimizer.step()
    
                    epoch_loss += loss.item()  # Accumulate loss for this epoch
                    progress_bar.set_postfix({"loss": loss.item()})  # Update progress bar with current loss
    
                scheduler.step()
    
                # Log loss as a metric after each epoch
                mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
    
            # Log and register the model with MLflow
            mlflow.pytorch.log_model(
                self.normflow, artifact_path="model", registered_model_name="MBP"
            )
    
            mlflow.set_tag("project", "MBP")
            mlflow.set_tag("input directory", f"{data_dir}")
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


    def process_catalog(self, data_dir, Nrealizations=100, return_distributions=False):
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
        batch_size=1
        print(f"Predicting dataset with {Nrealizations} realizations per object...")
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
            photoz_true[samp] =  meta[:,0,1]



            z_test = torch.randn(Nrealizations, self.input_dim).to(self.device)
            preds, _ = self.normflow(z_test, condition, rev=True)

            preds =  preds.detach().cpu().numpy()
            if self.predict_photoz:
                preds_photometry_all[samp] = preds[:,:-1] * max_norm.numpy()
                preds_all_photoz[samp] = preds[:,-1]
            else: 
                preds_photometry_all[samp] = preds * max_norm.numpy()
                

        if return_distributions:
            return preds_photometry_all, preds_all_photoz
        elif self.predict_photoz==True:
            photometric_preds_mean = np.nanmean(preds_photometry_all, axis=1)
            photometric_preds_err = np.nanstd(preds_photometry_all, axis=1)

            photoz_preds_mean = np.nanmean(preds_all_photoz, axis=1)
            photoz_preds_err = np.nanstd(preds_all_photoz, axis=1)

            return photometric_preds_mean, photometric_preds_err, photometry_true, photoz_preds_mean, photoz_preds_err, photoz_true
        else:
            photometric_preds_mean = np.nanmean(preds_photometry_all, axis=1)
            photometric_preds_err = np.nanstd(preds_photometry_all, axis=1)

            return photometric_preds_mean, photometric_preds_err, photometry_true

