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
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Assuming necessary imports for SBP_model_multExp, NF_model_MBP, etc.
sys.path.append('../SBP')
sys.path.append('../MBP')
sys.path.append('../bookkeeper')
sys.path.append('../data')
sys.path.append('../utils')

from SBP_models import SBP_model_multExp
from MBP_models import NF_model_MBP
from dataloader import create_dataloaders
from SBP import SBP
from dataset import DataSet
from load_image import load_image
class MBPz:
    def __init__(self, 
                 model_path_sbp, 
                 nexp=3,
                 verbose=True,
                 model_path_normflow=None,
                 save_path=None,
                 file_type='image',
                 flow_type='affine',
                 predict_photoz=True,
                 ntransformation=8):
        """
        Initializes the MBPz model, setting up paths, models, and hyperparameters.

        Arguments:
        - model_path_sbp (str): Path to the pre-trained SBP model checkpoint.
        - nexp (int): Number of exposure samples to generate. Default is 3.
        - verbose (bool): If True, print detailed logs during training. Default is True.
        - model_path_normflow (str or None): Path to a pre-trained normalizing flow model, if available. Default is None.
        - save_path (str or None): Directory to save the trained model checkpoints. Default is None.
        - file_type (str): Type of input files (e.g., 'image', 'csv'). Default is 'image'.
        - flow_type (str): Type of normalizing flow ('affine' or 'gaussianization'). Default is 'affine'.
        - predict_photoz (bool): Whether to predict photometric redshifts. Default is True.
        - ntransformation (int): Number of transformations in the normalizing flow. Default is 8.
        """
        print("Initializing MBPz model...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.file_type = file_type
        self.nexp = nexp
        self.ntransformation = ntransformation
        self.flow_type = flow_type
        self.predict_photoz = predict_photoz
        self.save_path = save_path
        self.verbose = verbose

        self.model_path_sbp=model_path_sbp
        
        # Initialize SBP model
        print("Loading SBP model...")
        checkpoint = torch.load(model_path_sbp, map_location=self.device)
        model_sbp = SBP_model_multExp(zp_calib=checkpoint['metadata']['model_metadata']['zp_calib'])
        model_sbp.load_state_dict(checkpoint['model_state_dict'])
        self.model_sbp = model_sbp.eval().to(self.device)  # Set the model to evaluation mode
        
    
        # Retrieve zp_calib from model metadata
        zp_calib = checkpoint['metadata']['model_metadata']['zp_calib']
        bands = checkpoint['metadata']['model_metadata']['bands']
        
        self.bands = bands
        self.nbands = len(bands)        
        # Initialize normalizing flow model
        print("Initializing normalizing flow model...")
        if predict_photoz:
            self.input_dim = self.nbands + 1
        else:
            self.input_dim = self.nbands
                        
        if self.flow_type == 'affine':
            self.normflow = NF_model_MBP(dim_inputSpace=self.input_dim)
        
        if model_path_normflow is not None:
            checkpoint = torch.load(model_path_normflow, map_location=self.device)
            self.normflow.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded pre-trained normalizing flow model.")
            self.batch_size=checkpoint['metadata']['model_metadata']['training_hyperparams']['batch_size']
        self.normflow = self.normflow.to(self.device)
        
        
        print("MBPz model initialized.")

    def train(self, data_dir, training_hyperparams):
        """
        Trains the normalizing flow model using the provided data.
    
        Arguments:
        None (all necessary parameters are initialized in the __init__ method).
        """
        print("Creating data loaders...")
        mlflow.autolog()  # Enable autologging of parameters, models, etc.
        
        loader_train, loader_val = create_dataloaders(data_dir, 
                                                      self.bands,
                                                      nexp=self.nexp,
                                                      batch_size=training_hyperparams['batch_size'],
                                                      file_type=self.file_type)
    
        optimizer = optim.Adam(self.normflow.parameters(), lr=training_hyperparams['learning_rate'])  
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    
        with mlflow.start_run() as run:
            mlflow.set_tag("project", "Flow4z:MBP")
            mlflow.set_tag("input directory", f"{data_dir}")
            mlflow.set_tag("Model_SBP", f"{self.model_path_sbp}")
            mlflow.set_tag("Model_MBP", f"{self.save_path}")
            mlflow.set_tag("predict_photoz", f"{self.predict_photoz}")
    
            # Log parameters for the run
            mlflow.log_param("learning_rate", training_hyperparams['learning_rate'])
            mlflow.log_param("batch_size", training_hyperparams['batch_size'])
            mlflow.log_param("nepochs", training_hyperparams['nepochs'])
            mlflow.log_param("nexp", self.nexp)
    
            mlflow.pytorch.log_model(self.normflow, "MBP-NF")
    
            # Register the model
            try:
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, "Flow4z:MBP")
                print(f"Model registered in the Model Registry under 'Flow4z:MBP'")
            except MlflowException as e:
                print(f"Error registering model: {e}")
    
            print(f"Starting training for {training_hyperparams['nepochs']} epochs...")
            
            # Training loop
            for epoch in range(training_hyperparams['nepochs']):
                epoch_loss = 0.0
                progress_bar = tqdm(loader_train, desc=f"Epoch {epoch + 1}/{training_hyperparams['nepochs']}", unit="epoch")
                for meta, data, max_norm in progress_bar:
                    optimizer.zero_grad()
    
                    if self.file_type == 'image':
                        features = torch.zeros(size=(len(data), self.nbands, 10))
                        flab = meta[:, :, 0, 1] / max_norm[:, :, 0]
                        for b in range(self.nbands):
                            f, feature = predict_flux(self.model_sbp, data[:, b, :, :].unsqueeze(1))
                            features[:, b, :] = feature.detach()
                        features = features.reshape(len(features), self.nbands * 10)
                        
                    elif self.file_type == 'features':
                        flab = meta[:, :, 0] / max_norm
                        features = data.reshape(len(data), self.nbands * 10)
                        
    
                    if self.predict_photoz:
                        input_nf = torch.cat((flab, meta[:, 0:1, 0]), dim=1)
                    else:
                        input_nf = flab
                        
    
                    if self.flow_type == 'affine':
                        z, log_jac_det = self.normflow(input_nf.to(self.device), features.to(self.device))
                        loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
                        loss = loss.mean()
                        
                    elif self.flow_type == 'gaussianization':
                        input_nf = torch.DoubleTensor(input_nf)
                        log_pdf, _, _ = self.normflow(input_nf.to(self.device), 
                                                      conditional_input=features.to(self.device))
                        loss = -log_pdf.mean()
    
                    loss.backward()
                    optimizer.step()
    
                    epoch_loss += loss.item()  # Accumulate loss for this epoch
                
                scheduler.step()
    
                # Log loss as a metric after each epoch
                mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
    
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{training_hyperparams['nepochs']}], Loss: {epoch_loss:.4f}")
           
    
            # Optionally log and register the final model
            mlflow.pytorch.log_model(self.normflow, "Flow4z_MBP_Final")
            print("Training completed.")

        return self.normflow


    def process_catalog(self,
                        data_dir, 
                        Nrealizations=100, 
                        return_distributions=False):
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
        print(f"Predicting dataset with {Nrealizations} realizations per object...")
        loader_test = create_dataloaders(path_data=data_dir,
                                         batch_size=1,
                                         nexp=self.nexp,
                                         test_size=1000,
                                         bands=self.bands,
                                         file_type=self.file_type)
        
        nobj = len(os.listdir(data_dir))
        preds_all = np.zeros((nobj, Nrealizations, self.input_dim))
        photometry_true = np.zeros(shape=(nobj,len(self.bands)))

        progress_bar = tqdm(loader_test, 
                            desc="Prediction Progress")
        
        for samp, (meta, features, max_norm) in enumerate(progress_bar):
            Niter = int(Nrealizations / self.batch_size)
            preds = np.zeros(shape=(Niter, self.batch_size, self.input_dim))
            
            features = features.reshape(len(features), self.nbands * 10) 
            condition = torch.tile(features, (self.batch_size, 1)).to(self.device)    

            photometry_true[samp] = meta[0,:,0] 

            # Generate predictions
            for ii in range(Niter):
                z_test = torch.randn(self.batch_size, self.input_dim).to(self.device)
                pred, _ = self.normflow(z_test, condition, rev=True)
                preds[ii] = pred.detach().cpu().numpy()

                
            preds = preds.reshape(Niter * self.batch_size, self.input_dim)
            preds_all[samp] = preds * max_norm.numpy()

            
        if return_distributions:
            return preds_all
        elif self.predict_photoz:                    
            photometric_preds_mean =  np.nanmean(photometric_preds, axis=1)
            photoz_preds_mean  = np.nanmean(photoz_preds, axis=1)
            return photometric_preds_mean, photoz_preds_mean
        else:
            photometric_preds_mean =  np.nanmean(preds_all, axis=1)
            return photometric_preds_mean, photometry_true

    def process_catalog_fromImage(self, data_dir,
                                  Nrealizations=100, 
                                  return_distributions=False):
        
        nobj = len(os.listdir(data_dir))
        preds_all = np.zeros((nobj, Nrealizations, self.input_dim))
        photometry_true = np.zeros(shape=(nobj,len(self.bands)))
        
        sbp = SBP(model_path=self.model_path_sbp)

        for i in tqdm(range(nobj), desc="Processing objects", unit="object"):
            stamps = torch.zeros(size=(len(self.bands), self.nexp, 60, 60))
            max_norms = np.zeros(shape=5)
            zps = torch.zeros(size=(len(self.bands), self.nexp))
            features = torch.zeros(size=(len(self.bands), 10))

            for ib, band in enumerate(self.bands):
                max_norm = 0
                for exp in range(self.nexp):
                    stamps[ib, exp], max_stamp, meta = load_image(data_dir, i, band, exp)
                    
                    max_norm += max_stamp
                max_norm = max_norm / self.nexp
                stamps[ib] = stamps[ib] / max_norm
                max_norms[ib] =max_norm

                f, feature = sbp.predict_flux(stamps[ib].unsqueeze(0).unsqueeze(0))
                
                features[ib, :] = feature.detach()

            features = features.reshape(1, self.nbands * 10) 
            condition = torch.tile(features, (self.batch_size, 1)).to(self.device) 
                         
            Niter = int(Nrealizations / self.batch_size)
            preds = np.zeros(shape=(Niter, self.batch_size, self.input_dim))
            # Generate predictions
            for ii in range(Niter):
                z_test = torch.randn(self.batch_size, self.input_dim).to(self.device)
                pred, _ = self.normflow(z_test, condition, rev=True)
                preds[ii] = pred.detach().cpu().numpy()


            preds = preds.reshape(Niter * self.batch_size, self.input_dim)
            preds_all[i] = preds * max_norms[None,:]
            #photometry_true[samp] = m[:, :, 0, 1].reshape(len(m) * len(self.bands)).detach().cpu().numpy()   
            print(max_norms)
            print(preds_all[i].mean(0))



        return preds_all.mean(1), photometry_true
