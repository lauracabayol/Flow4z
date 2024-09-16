#!/usr/bin/env python
import torch
from torch import nn

class SBP_model_multExp(nn.Module):
    def __init__(self, zp_calib=False):

        super().__init__()

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16, kernel_size = 9, padding = 4),nn.MaxPool2d(2), nn.BatchNorm2d(16), nn.ReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding = 4),nn.MaxPool2d(2), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(in_channels=32,out_channels=64, kernel_size = 5, padding = 3),nn.MaxPool2d(2), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding = 3),nn.MaxPool2d(2), nn.BatchNorm2d(32) ,nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, padding = 2),nn.MaxPool2d(2), nn.BatchNorm2d(16) ,nn.ReLU())
        
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(1,3)), 
            nn.AdaptiveAvgPool2d((10,10)),
            nn.ReLU())
        
        
        self.combine_linear = nn.Sequential(
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
            )
                
        
        if zp_calib:
            dim_linear=16*4*4+1
        else:
            dim_linear=16*4*4

        self.linear_features = nn.Sequential(nn.Linear(dim_linear, 200),
                                             nn.Dropout(0.02),
                                             nn.ReLU(),
                                            nn.Linear(200, 50),
                                             nn.Dropout(0.02),
                                             nn.ReLU(),
                                            nn.Linear(50, 10),
                                             nn.Dropout(0.02),
                                             nn.ReLU())

        self.ll_alpha = nn.Sequential(nn.Linear(10, 5))
        self.ll_sigma = nn.Sequential(nn.Linear(10, 5))
        self.ll_mean = nn.Sequential(nn.Linear(10, 5))
        
        
    def forward(self, img, zp=False):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        features = torch.Tensor().to(device)
        
        for e in range(img.shape[2]):
            x = self.encoder1(img[:,:,e,:,:])
            x = x.view(len(x),-1) #convert to 1D array
            if zp !=False:
                x = torch.cat((x, zp[:,e].unsqueeze(1)), 1)
            feature = self.linear_features(x)
            features = torch.cat((features,feature),1)
                                    
        if img.shape[2] > 1:
            features = features.view(len(features),1, img.shape[2], 10)
            features = self.combine(features)

            features = features.view(len(features),-1) 
            features = self.combine_linear(features)
            
            
        flux = self.ll_mean(features)
        logalpha = self.ll_alpha(features)
        logsig = self.ll_sigma(features)

        logalpha = logalpha - torch.logsumexp(logalpha, 1)[:,None]
        
        return flux,logalpha,logsig, features   



class SBP_model_singExp(nn.Module):
    def __init__(self):

        super().__init__()

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16, kernel_size = 9, padding = 4),nn.MaxPool2d(2), nn.BatchNorm2d(16), nn.ReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding = 4),nn.MaxPool2d(2), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(in_channels=32,out_channels=64, kernel_size = 5, padding = 3),nn.MaxPool2d(2), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding = 3),nn.MaxPool2d(2), nn.BatchNorm2d(32) ,nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, padding = 2),nn.MaxPool2d(2), nn.BatchNorm2d(16) ,nn.ReLU())

        self.ll1 = nn.Sequential(nn.Linear(16*4*4, 200),nn.Dropout(0.02),nn.ReLU())
        self.ll2 = nn.Sequential(nn.Linear(200, 50),nn.Dropout(0.02),nn.ReLU())
        self.ll3 = nn.Sequential(nn.Linear(50, 10),nn.Dropout(0.02),nn.ReLU())
        
        self.ll_alpha = nn.Sequential(nn.Linear(10, 5))
        self.ll_sigma = nn.Sequential(nn.Linear(10, 5))
        self.ll_mean = nn.Sequential(nn.Linear(10, 5))
        
        
    def forward(self, img, mode='train'):

        x = self.encoder1(img, zp=None)
        x = x.view(len(x),-1) #convert to 1D array

        #additional input information (band, CCD galaxy coordinates)
        if zp is not None:
            x = torch.cat((x, zp), 1)
        x = self.ll1(x)
        x = self.ll2(x)
        x = self.ll3(x)
        flux = self.ll_mean(x)
        logalpha = self.ll_alpha(x)
        logsig = self.ll_sigma(x)

        logalpha = logalpha - torch.logsumexp(logalpha, 1)[:,None]
        
        if mode == 'train':
            return flux,logalpha,logsig   
        else:
            return flux,logalpha,logsig, x
