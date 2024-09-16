import torch
from torch import nn, optim
import numpy as np
import datetime
import time
import os
import sys
from tqdm import tqdm  # Progress bar

# Assuming necessary imports for SBP_model_multExp, NF_model_MBP, etc.
sys.path.append('../SBP')
sys.path.append('../MBP')
sys.path.append('../bookkeeper')
sys.path.append('../data')
sys.path.append('../utils')

from SBP import predict_flux
from SBP_models import SBP_model_multExp
from MBP_models import NF_model_MBP
from dataloader import create_dataloaders

class MBPz:
    def __init__(self, 
                 model_path_sbp, 
                 data_dir_path,
                 nepochs=100,
                 lr=1e-3,
                 batch_size=100,
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
        - data_dir_path (str): Path to the directory containing the dataset.
        - nepochs (int): Number of epochs for training. Default is 100.
        - lr (float): Learning rate for the optimizer. Default is 1e-3.
        - batch_size (int): Number of samples per batch. Default is 100.
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
        self.data_dir_path = data_dir_path
        self.batch_size = batch_size
        self.ntransformation = ntransformation
        self.flow_type = flow_type
        self.predict_photoz = predict_photoz
        self.save_path = save_path
        self.verbose = verbose
        
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
            self.normflow.load_state_dict(torch.load(model_path_normflow, map_location=self.device))
            print("Loaded pre-trained normalizing flow model.")
        self.normflow = self.normflow.to(self.device)
        
        # Hyperparameters
        self.nepochs = nepochs
        self.lr = lr

        print("MBPz model initialized.")

    def _train_model(self):
        """
        Trains the normalizing flow model using the provided data.

        Arguments:
        None (all necessary parameters are initialized in the __init__ method).
        """
        print("Creating data loaders...")
        loader_train, loader_val = create_dataloaders(self.data_dir_path, 
                                                      self.bands,
                                                      nexp=self.nexp,
                                                      batch_size=self.batch_size,
                                                      file_type=self.file_type)
        
        optimizer = optim.Adam(self.normflow.parameters(), lr=self.lr)  
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        print(f"Starting training for {self.nepochs} epochs...")
        for epoch in tqdm(range(self.nepochs), desc="Training Progress"):
            epoch_loss = 0.0  # To accumulate loss for the epoch
            for meta, data, max_norm in loader_train:
                optimizer.zero_grad()
                if self.file_type=='image':
                    features = torch.zeros(size=(len(data), self.nbands, 10))
                    flab = meta[:, :, 0, 1] / max_norm[:, :, 0]
                    for b in range(self.nbands):
                        f, feature = predict_flux(self.model_sbp, data[:, b, :, :].unsqueeze(1))
                        features[:, b, :] = feature.detach()
                    features = features.reshape(len(features), self.nbands * 10) 
                elif self.file_type=='features':
                    optimizer.zero_grad()
                    flab = meta[:, :, 1] / max_norm
                    features = data.reshape(len(data), self.nbands * 10) 
            
                if self.predict_photoz:
                    input_nf = torch.cat((flab, meta[:, 0:1, 1]), dim=1) 
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
                
                epoch_loss += loss.item()  # Accumulate loss for reporting

            scheduler.step()

            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.nepochs}], Loss: {epoch_loss:.4f}")
   
            
                
            self.save_model()

        print("Training completed.")

    def save_model(self):
        """
        Saves the model's state dictionary and metadata to the specified save path.

        """
        print(f"Saving model.")
        model_state_dict = self.normflow.state_dict()

        # Save metadata
        metadata = {
            'nepochs': self.nepochs,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'bands': self.bands,
            'file_type': self.file_type,
            'flow_type': self.flow_type,
            'predict_photoz': self.predict_photoz,
            'ntransformation': self.ntransformation,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save({
            'model_state_dict': model_state_dict,
            'metadata': metadata
        }, self.save_path)

    def predict_dataset(self,
                        path_data, 
                        nexp=3,
                        Nrealizations=100, 
                        return_features=False, 
                        return_distributions=True):
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
        print(f"Predicting dataset with {Nrealizations} realizations per object...")
        loader_test = create_dataloaders(path_data=path_data,
                                         batch_size=1,
                                         nexp=nexp,
                                         test_size=0,
                                         bands=self.bands,
                                         file_type=self.file_type)
        
        nobj = len(os.listdir(self.data_dir_path))
        preds_all = np.zeros((nobj, Nrealizations, self.input_dim))
        
        for samp, (meta, features, max_norm) in enumerate(tqdm(loader_test, desc="Prediction Progress")):
            Niter = int(Nrealizations / self.batch_size)
            preds = np.zeros(shape=(Niter, self.batch_size, self.input_dim))
            
            features = features.reshape(len(features), self.nbands * 10) 
            condition = torch.tile(features, (self.batch_size, 1)).to(self.device)            

            # Generate predictions
            for ii in range(Niter):
                z_test = torch.randn(self.batch_size, self.input_dim).to(self.device)
                pred, _ = self.normflow(z_test, condition, rev=True)
                preds[ii] = pred.detach().cpu().numpy()
                
            preds = preds.reshape(Niter * self.batch_size, self.input_dim)
            preds_all[samp] = preds
            
            photometric_preds = preds[:, :self.nbands]
            photoz_preds = preds[:, self.nbands:]
            
            photometric_preds_mean = np.nanmean(photometric_preds, axis=0)
            photoz_preds_mean  = np.nanmean(photoz_preds, axis=0)
                
        if return_features:
            return photometric_preds_mean, photoz_preds_mean, features
        if return_distributions:
            return preds_all
        if return_features and return_distributions:
            return photometric_preds, photoz_preds, features
        else:
            return photometric_preds_mean, photoz_preds_mean
