#!/usr/bin/env python
import torch
from torch import nn

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def NF_model_MBP(dim_inputSpace=5):


    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, 64),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, dims_out),
        )

    # Initialize the cINN model
    emulator = Ff.SequenceINN(dim_inputSpace)

    # Append AllInOneBlocks to the cINN model based on the specified number of layers
    for l in range(8):
        emulator.append(
            Fm.AllInOneBlock,
            cond=[i for i in range(100)],
            cond_shape=[10*5],
            subnet_constructor=subnet_fc,
        )

    return emulator

def NF_model_MBPz(dim_inputSpace=6, 
                  dim_CondSpace=100, 
                  predict_photoz=True,
                  ntransformation=8):


    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, 64),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, dims_out),
        )


    # Initialize the cINN model
    nf = Ff.SequenceINN(dim_inputSpace)
    if predict_photoz:
        dim_feat = dim_inputSpace-1
    else:
        dim_feat = dim_inputSpace
        
    # Append AllInOneBlocks to the cINN model based on the specified number of layers
    for l in range(ntransformation):
        nf.append(
            Fm.AllInOneBlock,
            cond=[i for i in range(dim_CondSpace)],
            cond_shape=[10*dim_feat],
            subnet_constructor=subnet_fc,
        )

    return nf