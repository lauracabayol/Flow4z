#import modules
import numpy as np
import sys
from scipy.stats import norm
import os
import json
#custom modules
sys.path.append('../data')
from dataset import DataSet
from load_image import load_image
sys.path.append('/.')

from SBP_models import SBP_model_multExp

#torch modules
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm  


def train_SBP(model, 
              loader_train, 
              nepochs, 
              nbands=5,
              nexp=3,
              zp_calib=None):
    """
    Train and test a neural network on a given dataset.

    Args:
        model: Neural network model to be trained.
        loader_train: DataLoader for the training data.
        nepochs (int): Number of training epochs.
        nbands (int): Number of photometric bands.
        zp_calib (float, optional): Zero-point calibration value. If None, zero-point calibration is not applied.

    Returns:
        model: Trained neural network model.
    """
    # Define network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define training optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    # Training loop
    for epoch in range(nepochs):
        # Initialize tqdm progress bar for this epoch
        progress_bar = tqdm(loader_train, desc=f"Epoch {epoch + 1}/{nepochs}", unit="epoch")

        for meta, stamp, max_norm in progress_bar:
            
            optimizer.zero_grad()

            # Prepare data
            meta = meta.squeeze(1)
            z, lab, zp = meta[:, :, 0], meta[:, :, 1], meta[:, :, 2]
            lab = lab / max_norm
            stamp = stamp.reshape(len(stamp) * nbands, stamp.shape[2], 60, 60).unsqueeze(1).float()
            
            lab = lab[:,:,0].reshape(len(lab) * nbands).unsqueeze(1)

            if zp_calib:
                zp = zp * torch.normal(1, zp_calib/100, size=zp.shape)
                zp = zp.reshape(len(zp) * nbands, nexp)
            
            # Run network
            if zp_calib:
                flux, logalpha, logsig, _ = model(stamp.to(device), zp.to(device))
            else:
                flux, logalpha, logsig, _ = model(stamp.to(device))

            # Ensure stability of logsig
            logsig = torch.clamp(logsig, -6, 6)
            sig = torch.exp(logsig)

            # Loss function
            log_prob = logalpha - 0.5 * (flux - lab.to(device)).pow(2) / sig.pow(2) - logsig
            log_prob = torch.logsumexp(log_prob, 1)
            loss = -log_prob.mean()

            # Backpropagate
            loss.backward()
            optimizer.step()

            # Update tqdm progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
            
        # Update learning rate scheduler
        scheduler.step()

        # Print training loss for the epoch
        print(f'Epoch {epoch + 1}/{nepochs} completed. Median loss: {np.median(loss.detach().cpu().numpy())}')

    return model


def predict_flux(model, 
                 image, 
                 return_pdf=False,
                 zp_calib=None):
    """
    Predict flux using a trained neural network model.

    Args:
        model (SBP_model): Trained neural network model.
        image (torch.Tensor): Input image tensor.

    Returns:
        numpy.ndarray: Predicted flux values.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set the model to evaluation mode and move it to the device
    model = model.eval().to(device)

    # Run the model on the input image
    if zp_calib != None:
        flux, logalpha, logsig, features = model(image.to(device), zp_calib.to(device))
    else:
        flux, logalpha, logsig, features = model(image.to(device))

    # Calculate the predicted flux by summing over the channels
    f = (torch.exp(logalpha) * flux).sum(1)
    
    if return_pdf:
        alphas = torch.exp(logalpha).detach().cpu().numpy()
        sig = torch.exp(logsig).detach().cpu().numpy()
        alphas = alphas / alphas.sum(1)[:,None]
        flux = flux.detach().cpu().numpy()
        
        f_axis = np.concatenate((np.arange(-10,100,0.1),np.arange(100,500,1)),0)

        pdfs =  alphas[:,:,None] * norm.pdf(f_axis,loc =flux[:,:,None],scale =sig[:,:,None])
        pdfs = pdfs.sum(1)
        pdfs_norm = pdfs/pdfs.sum(1)[:,None]
        
        return f.detach().cpu().numpy(), features, pdfs_norm

    return f.detach().cpu().numpy(), features







def get_features(data_dir,
                 model_path,
                 nexp=3):
    """
    Extract features from the image data using a neural network model.

    Args:
        data_dir (str): Path to the dataset directory.
        model_path (str): Path to the saved neural network model.
        bands_survey (str): The survey type, either 'PAUS' or 'CFHT'.
        add_calib_error (bool): Whether to add calibration error to the ZP (default: False).
        nexp (int): Number of exposures (default: 3).
    """
    
    # Load the model and its metadata
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    
    model_sbp = SBP_model_multExp(zp_calib=checkpoint['metadata']['model_metadata']['zp_calib'])
    model_sbp.load_state_dict(checkpoint['model_state_dict'])
    model_sbp = model_sbp.eval().to(device)  # Set the model to evaluation mode

    # Retrieve zp_calib from model metadata
    zp_calib = checkpoint['metadata']['model_metadata']['zp_calib']
    bands = checkpoint['metadata']['model_metadata']['bands']
    
    nobj = len(os.listdir(data_dir))  # Number of objects in the dataset
    
    # Loop through each object in the dataset
    for i in range(nobj):
        stamps = torch.zeros(size=(len(bands), nexp, 60, 60))  # Initialize tensor for image data
        zps = torch.zeros(size=(len(bands), nexp))  # Initialize tensor for ZP data
        features = torch.zeros(size=(len(bands), 10))  # Initialize tensor for extracted features

        # Loop through each band
        for ib, band in enumerate(bands):
            max_norm = 0
            # Loop through each exposure
            for exp in range(nexp):
                stamps[ib, exp], max_stamp, meta = load_image(data_dir, i, band, exp)
                zps[ib, exp] = meta[0, 2]  # Store the zero-point calibration
                max_norm += max_stamp  # Accumulate max pixel values
            max_norm = max_norm / nexp  # Average max pixel values
            stamps[ib] = stamps[ib] / max_norm  # Normalize the image data
            
            # Optionally add calibration error
            if zp_calib!= False:
                zps = zps * torch.normal(1, zp_calib, size=zps.shape)

            # Predict flux and extract features using the model
            if zp_calib:
                f, feature = predict_flux(model_sbp, 
                                          stamps[ib].unsqueeze(0).unsqueeze(0),
                                          zp_calib=zps[ib].unsqueeze(0))
            else:
                f, feature = predict_flux(model_sbp, 
                                          stamps[ib].unsqueeze(0).unsqueeze(0))
                
            features[ib, :] = feature.detach()  # Store extracted features
        
        # Save the extracted features
        features_path = f'{data_dir}/data_{i}/features_{i}.npy'
        np.save(features_path, features)