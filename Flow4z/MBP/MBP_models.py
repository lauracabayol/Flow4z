"""Models for the MBP (Model-Based Photometry) module."""
import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, dims_out),
    )

def NF_model_MBP(dim_inputSpace=5):
    emulator = Ff.SequenceINN(dim_inputSpace)
    for _ in range(8):
        emulator.append(
            Fm.AllInOneBlock,
            cond=list(range(100)),
            cond_shape=[50],
            subnet_constructor=subnet_fc,
        )
    return emulator

def NF_model_MBPz(dim_inputSpace=6, dim_CondSpace=100, predict_photoz=True, ntransformation=8):
    nf = Ff.SequenceINN(dim_inputSpace)
    dim_feat = dim_inputSpace - 1 if predict_photoz else dim_inputSpace
    
    for _ in range(ntransformation):
        nf.append(
            Fm.AllInOneBlock,
            cond=list(range(dim_CondSpace)),
            cond_shape=[10 * dim_feat],
            subnet_constructor=subnet_fc,
        )
    return nf