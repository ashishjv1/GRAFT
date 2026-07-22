
import torch
import numpy as np
import warnings

# Suppress specific NumPy 2.0 warnings for backward compatibility
warnings.filterwarnings("ignore", message="__array_wrap__ must accept context and return_scalar arguments")

def calnorm(idxgrads, fgrads):
    # Keep everything in PyTorch to avoid NumPy 2.0 warnings
    ss_grad = torch.transpose(idxgrads.clone().detach().cpu(), 0, 1)
    b_ = fgrads.sum(dim=0).detach().cpu()
    
    # Use PyTorch's pinverse instead of NumPy
    pinverse = torch.pinverse(ss_grad.float())
    x = torch.matmul(pinverse, b_.float())
    
    # Calculate residual norm
    norm_residual = torch.norm(torch.matmul(ss_grad.float(), x) - b_.float())
    return norm_residual
